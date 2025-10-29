#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import sounddevice as sd
import queue
import threading
import time
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ====================== 配置 ======================
MODEL_PATH = "/data/huangtianle/ASR_train/whisper-small-finetuned-V1"
SILENCE_THRESHOLD = 0.02           # 提高阈值，减少误触发
MIN_SPEECH_SEC = 0.5               # 最小语音长度
MAX_SPEECH_SEC = 15.0              # 最大语音长度
SILENCE_DURATION_SEC = 0.8         # 静音 >0.8s 才切句
LANGUAGE = "zh"
# =================================================

print("正在加载模型...")
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
model.eval()
if torch.cuda.is_available():
    model = model.cuda()

RATE = 16000
audio_queue = queue.Queue()
stop_event = threading.Event()

class SpeechRecognizer(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.buffer = np.array([], dtype=np.float32)
        self.silence_start = None
        self.speech_start = None
        self.last_result = ""
        self.total_time = 0.0

    def run(self):
        frame_size = int(0.1 * RATE)
        min_samples = int(MIN_SPEECH_SEC * RATE)
        max_samples = int(MAX_SPEECH_SEC * RATE)
        silence_frames = int(SILENCE_DURATION_SEC * RATE)

        while not stop_event.is_set():
            try:
                chunk = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self.buffer = np.concatenate([self.buffer, chunk])
            self.total_time += len(chunk) / RATE

            if len(self.buffer) < frame_size:
                continue

            recent = self.buffer[-frame_size:]
            energy = np.mean(np.abs(recent))

            current_pos = len(self.buffer)

            if energy > SILENCE_THRESHOLD:
                if self.speech_start is None:
                    self.speech_start = current_pos - frame_size
                self.silence_start = None
            else:
                if self.speech_start is not None and self.silence_start is None:
                    self.silence_start = current_pos

                # 静音足够长，且有足够语音
                if (self.silence_start and 
                    current_pos - self.silence_start > silence_frames and
                    self.silence_start - self.speech_start > min_samples):

                    segment = self.buffer[self.speech_start:self.silence_start]
                    remaining = self.buffer[self.silence_start:]

                    if len(segment) <= max_samples:
                        text = self.transcribe(segment)
                        if text and len(text) > 1 and text != self.last_result:
                            start = self.total_time - len(remaining)/RATE - len(segment)/RATE
                            end = self.total_time - len(remaining)/RATE
                            print(f"[{start:6.2f}s - {end:6.2f}s] {text}")
                            self.last_result = text

                    # 重置
                    self.buffer = remaining
                    self.speech_start = None
                    self.silence_start = None

            # 防止 buffer 过长
            if len(self.buffer) > RATE * 30:
                self.buffer = self.buffer[-RATE*10:]

    def transcribe(self, audio_np):
        try:
            inputs = processor(audio_np, sampling_rate=RATE, return_tensors="pt")
            if torch.cuda.is_available():
                inputs.input_features = inputs.input_features.cuda()

            with torch.no_grad():
                ids = model.generate(
                    inputs.input_features,
                    max_length=448,
                    num_beams=1,
                    language=LANGUAGE,
                    task="transcribe",
                    no_repeat_ngram_size=2,  # 防止重复
                )

            text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            return text if len(text) > 1 else None
        except:
            return None

# 音频回调
def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio:", status)
    audio_queue.put(indata.copy()[:, 0])

# 主函数
def main():
    recognizer = SpeechRecognizer()
    recognizer.start()

    print(f"\n实时语音识别启动 (VAD + 防幻觉)")
    print(f"语言: 中文 | 最小语音: {MIN_SPEECH_SEC}s | 静音切句: {SILENCE_DURATION_SEC}s")
    print("请说：你好语音识别\n")

    try:
        with sd.InputStream(samplerate=RATE, channels=1, dtype='float32',
                            blocksize=int(RATE * 0.1), callback=audio_callback):
            while not stop_event.is_set():
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n停止。")
    finally:
        stop_event.set()

if __name__ == "__main__":
    main()