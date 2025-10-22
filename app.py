from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import os
import subprocess
import json
import uuid
from werkzeug.utils import secure_filename
import zipfile
import shutil
from pathlib import Path
from threading import Thread
import queue
import time
import signal
import sys

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:8082"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "ASR_train_models")
UPLOAD_DIR = os.path.join(BASE_DIR, "Uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
VITS_BASE = os.path.join(BASE_DIR, 'VITS-fast-fine-tuning', 'dataset')
CLEANERS_FILE = os.path.join(BASE_DIR, 'VITS-fast-fine-tuning', 'text', 'cleaners.py')
CONFIG_DIR = os.path.join(BASE_DIR, 'VITS-fast-fine-tuning', 'configs')
ALLOWED_EXTENSIONS = {'zip'}
OFFLINE_MODEL_DIR = os.path.join(BASE_DIR, "models", "ASR_models")  # 新增：离线模型目录
OFFLINE_UPLOAD_DIR = os.path.join(BASE_DIR, "Uploads", "offline_audio")  # 新增：离线音频上传目录
ALLOWED_AUDIO_EXTENSIONS = {'wav'}  # 新增：允许的音频文件扩展名


os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(OFFLINE_UPLOAD_DIR, exist_ok=True)  # 新增：创建离线音频目录

training_process = None
asr_training_process = None
log_queue = queue.Queue()
asr_log_queue = queue.Queue()
training_thread = None
asr_training_thread = None
process_killed = False
asr_process_killed = False

def allowed_file(filename, allowed_extensions=ALLOWED_EXTENSIONS):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def extract_zip(zip_path, extract_to):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        items = os.listdir(extract_to)
        if len(items) == 1 and os.path.isdir(os.path.join(extract_to, items[0])):
            inner_dir = os.path.join(extract_to, items[0])
            for item in os.listdir(inner_dir):
                shutil.move(os.path.join(inner_dir, item), extract_to)
            shutil.rmtree(inner_dir)
        return True
    except Exception as e:
        return str(e)

@app.route('/api/models', methods=['GET'])
def get_models():
    try:
        models = [f for f in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, f))]
        return jsonify({"models": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({"error": "没有上传文件"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_DIR, filename)
        file.save(upload_path)
        extract_dir = os.path.join(UPLOAD_DIR, os.path.splitext(filename)[0])
        os.makedirs(extract_dir, exist_ok=True)
        result = extract_zip(upload_path, extract_dir)
        if result is True:
            return jsonify({"message": "数据集上传并解压成功", "dataset_path": extract_dir})
        else:
            return jsonify({"error": f"解压失败: {result}"}), 500
    return jsonify({"error": "文件格式不支持"}), 400

@app.route('/api/upload_vits_dataset', methods=['POST'])
def upload_vits_dataset():
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    filename = secure_filename(file.filename)
    if not filename.lower().endswith('.zip'):
        return jsonify({'error': '文件格式不支持，需为 zip'}), 400

    name = os.path.splitext(filename)[0]
    upload_path = os.path.join(UPLOAD_DIR, filename)
    file.save(upload_path)

    tmp_extract = os.path.join(UPLOAD_DIR, f"{name}_tmp")
    os.makedirs(tmp_extract, exist_ok=True)

    result = extract_zip(upload_path, tmp_extract)
    if result is not True:
        shutil.rmtree(tmp_extract, ignore_errors=True)
        return jsonify({'error': f'解压失败: {result}'}), 500

    target_base = VITS_BASE
    os.makedirs(target_base, exist_ok=True)

    for entry in os.listdir(target_base):
        path = os.path.join(target_base, entry)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except Exception as e:
            print(f"警告: 无法删除 {path}: {e}")

    target_dir = os.path.join(target_base, name)
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    for item in os.listdir(tmp_extract):
        s = os.path.join(tmp_extract, item)
        d = os.path.join(target_dir, item)
        shutil.move(s, d)
    shutil.rmtree(tmp_extract, ignore_errors=True)

    subfolders = sorted([d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))])

    debug_info = {
        'top_level': sorted(os.listdir(target_dir)),
        'subfolders': {}
    }

    train_lines = []
    val_lines = []

    for idx, sub in enumerate(subfolders):
        sub_path = os.path.join(target_dir, sub)
        wavs = sorted([f for f in os.listdir(sub_path) if f.lower().endswith('.wav')])
        txts = sorted([f for f in os.listdir(sub_path) if f.lower().endswith('.txt') or f.lower().endswith('.lab') or f.lower().endswith('.text')])
        debug_info['subfolders'][sub] = {'wav_count': len(wavs), 'txt_count': len(txts)}

        pairs = []
        missing_txt = []
        for wav in wavs:
            base = os.path.splitext(wav)[0]
            txt_path = os.path.join(sub_path, base + '.txt')
            if not os.path.exists(txt_path):
                for ext in ['.lab', '.text']:
                    if os.path.exists(os.path.join(sub_path, base + ext)):
                        txt_path = os.path.join(sub_path, base + ext)
                        break
            if not os.path.exists(txt_path):
                missing_txt.append(wav)
                continue

            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                text_line = lines[1] if len(lines) > 1 else (lines[0] if lines else '')
            except Exception:
                text_line = ''
            rel_path = f"./dataset/{name}/{sub}/{wav}"
            pairs.append((rel_path, str(idx), text_line))

        debug_info['subfolders'][sub]['missing_txt'] = missing_txt

        n = len(pairs)
        if n == 0:
            continue
        split = int(n * 0.9)
        if split < 1:
            split = n - 1 if n > 1 else 1
        for i, item in enumerate(pairs):
            line = '|'.join(item)
            if i < split:
                train_lines.append(line)
            else:
                val_lines.append(line)

    if not train_lines and not val_lines:
        return jsonify({'error': '未找到有效的 wav-txt 配对样本', 'debug': debug_info}), 400

    train_file = os.path.join(target_base, f"{name}_train.txt")
    val_file = os.path.join(target_base, f"{name}_val.txt")
    try:
        with open(train_file, 'w', encoding='utf-8') as f:
            for ln in train_lines:
                f.write(ln + '\n')
        with open(val_file, 'w', encoding='utf-8') as f:
            for ln in val_lines:
                f.write(ln + '\n')
    except Exception as e:
        return jsonify({'error': f'写入 train/val 文件失败: {str(e)}', 'debug': debug_info}), 500

    def generate_symbols(train_file_path, val_file_path):
        symbols = set()
        for file_path in [train_file_path, val_file_path]:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('|')
                        if len(parts) < 3:
                            continue
                        text = parts[2].strip()
                        if not text:
                            continue
                        tokens = text.split()
                        for t in tokens:
                            if t:
                                symbols.add(t)
            except Exception:
                continue
        return sorted(list(symbols))

    symbols = generate_symbols(train_file, val_file)
    symbols_file = os.path.join(target_base, f"{name}_symbols.txt")
    try:
        with open(symbols_file, 'w', encoding='utf-8') as f:
            f.write('[\n')
            for i, s in enumerate(symbols):
                comma = ',' if i != len(symbols) - 1 else ''
                f.write(f'  "{s}"{comma}\n')
            f.write(']\n')
    except Exception as e:
        return jsonify({'error': f'写入 symbols 文件失败: {str(e)}', 'debug': debug_info}), 500

    return jsonify({
        'message': '上传并处理成功',
        'dataset_name': name,
        'dataset_dir': os.path.abspath(target_dir),
        'train_file': os.path.abspath(train_file),
        'val_file': os.path.abspath(val_file),
        'symbols_file': os.path.abspath(symbols_file),
        'debug': debug_info
    })

@app.route('/api/confirm_vits_params', methods=['POST'])
def confirm_vits_params():
    data = request.get_json()
    if not data:
        return jsonify({'error': '缺少参数数据'}), 400

    training_files = data.get('data', {}).get('training_files')
    if not training_files:
        return jsonify({'error': '缺少 training_files'}), 400

    try:
        dataset_name = os.path.splitext(os.path.basename(training_files))[0]
        dataset_name = dataset_name.replace('_train', '').replace('_val', '')
        if not dataset_name:
            return jsonify({'error': '无法从 training_files 提取有效的数据集名称'}), 400
    except Exception as e:
        return jsonify({'error': f'无法提取数据集名称: {str(e)}'}), 400

    try:
        config_path = os.path.join(BASE_DIR, 'VITS-fast-fine-tuning', 'configs', 'modified_finetune_speaker.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        config['train'].update(data['train'])
        config['data']['training_files'] = data['data']['training_files']
        config['data']['validation_files'] = data['data']['validation_files']
        config['data']['text_cleaners'] = data['data']['text_cleaners']
        config['data']['n_speakers'] = data['data']['n_speakers']
        config['model']['gin_channels'] = data['model']['gin_channels']
        config['model']['speakers'] = data['model']['speakers']
        symbols_file = data.get('symbols_file')
        if not os.path.exists(symbols_file):
            return jsonify({'error': f'符号文件 {symbols_file} 不存在'}), 400
        with open(symbols_file, 'r', encoding='utf-8') as f:
            config['symbols'] = json.load(f)
        config['preserved'] = data.get('preserved', 2)

        config_filename = f"{dataset_name}_finetune_speaker.json"
        config_path = os.path.join(CONFIG_DIR, config_filename)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        text_cleaners = data['data']['text_cleaners'][0].replace('_cleaners', '')
        symbols = config['symbols']
        pinyin_func = 'pc.characters_to_jyutping(text)' if text_cleaners.lower() == 'yueyu' else 'pc.pinyin(text, style=pc.Style.TONE)'
        pinyin_var = 'jyutping_list' if text_cleaners.lower() == 'yueyu' else 'pinyin_list'
        cleaners_content = f"""# BEGIN_CUSTOM_CLEANERS
{text_cleaners.upper()}_SYMBOLS = set({json.dumps(symbols, ensure_ascii=False)})

def {text_cleaners}_cleaners(text: str) -> str:
    import re
    import pypinyin as pc
    text = re.sub(r'[^\\u4e00-\\u9fff]', ' ', text)
    {pinyin_var} = {pinyin_func}

    phones = []
    for py in {pinyin_var}:
        py = py[1] if isinstance(py, tuple) else py[0]
        if py and py.lower() in {text_cleaners.upper()}_SYMBOLS:
            phones.append(py.lower())
        else:
            phones.append('<unk>')
    
    return ' '.join(phones).strip() or '<unk>'
# END_OF_CUSTOM_CLEANERS
"""

        original_content = ""
        if os.path.exists(CLEANERS_FILE):
            with open(CLEANERS_FILE, 'r', encoding='utf-8') as f:
                original_content = f.read()

        begin_marker = "# BEGIN_CUSTOM_CLEANERS"
        end_marker = "# END_OF_CUSTOM_CLEANERS"
        new_content = original_content

        if begin_marker in original_content and end_marker in original_content:
            start_idx = original_content.index(begin_marker)
            end_idx = original_content.index(end_marker) + len(end_marker)
            new_content = original_content[:start_idx] + cleaners_content + original_content[end_idx:]
        else:
            new_content = original_content.rstrip() + "\n\n" + cleaners_content

        with open(CLEANERS_FILE, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return jsonify({
            'message': '参数确认成功，cleaners.py 已更新',
            'dataset_name': dataset_name,
            'config_path': os.path.abspath(config_path)
        })
    except Exception as e:
        return jsonify({'error': f'参数处理失败: {str(e)}'}), 500

@app.route('/api/train_vits', methods=['POST'])
def train_vits_model():
    global training_process, process_killed, training_thread
    data = request.get_json()
    model_save_path = data.get('model_save_path')
    config_path = data.get('config_path')
    preserved = data.get('preserved', 2)

    if not model_save_path:
        return jsonify({"error": "缺少 model_save_path"}), 400
    if not config_path:
        return jsonify({"error": "缺少 config_path"}), 400
    if not os.path.exists(config_path):
        return jsonify({"error": "配置文件不存在，请先确认参数"}), 400

    conda_env_path = os.path.join(os.path.expanduser("~"), "anaconda3", "envs", "VITS_train_env")
    python_executable = os.path.join(conda_env_path, "bin", "python")

    if not os.path.exists(python_executable):
        error_msg = f"Python executable not found: {python_executable}"
        with open(os.path.join(BASE_DIR, "train_log.txt"), 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {error_msg}\n")
        return jsonify({"error": error_msg}), 500

    cmd = [
        python_executable,
        os.path.join(BASE_DIR, "VITS-fast-fine-tuning", "finetune_speaker_v3.py"),
        "-m", model_save_path,
        "--drop_speaker_embed", "True",
        "-c", config_path,
        "--preserved", str(preserved)
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PATH"] = f"{os.path.join(conda_env_path, 'bin')}:{env['PATH']}"
    env["LD_LIBRARY_PATH"] = f"{os.path.join(conda_env_path, 'lib')}:/usr/lib/x86_64-linux-gnu:{env.get('LD_LIBRARY_PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    log_file = os.path.join(BASE_DIR, "train_log.txt")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Executing command: {' '.join(cmd)}\n")
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Environment PATH: {env['PATH']}\n")
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Environment LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}\n")
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Environment CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}\n")
        sys.stdout.flush()

    def run_training():
        global training_process, process_killed
        try:
            test_cmd = [python_executable, "-c", "import sys, torch, torio; print('Test: Python version: %s, CUDA: %s, FFmpeg: %s' % (sys.version, torch.cuda.is_available(), torio._extension._FFMPEG_EXT_LOADED)); sys.stdout.flush()"]
            test_process = subprocess.run(
                test_cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test command STDOUT: {test_process.stdout}\n")
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test command STDERR: {test_process.stderr}\n")
                if test_process.returncode != 0:
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test command failed with exit code {test_process.returncode}\n")
            log_queue.put(f"Test command output: {test_process.stdout.strip()}")
            if test_process.stderr:
                log_queue.put(f"Test command error: {test_process.stderr.strip()}")

            training_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=os.path.join(BASE_DIR, "VITS-fast-fine-tuning")
            )
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training process started with PID: {training_process.pid}\n")
                sys.stdout.flush()

            from select import select
            while training_process.poll() is None:
                rlist, _, _ = select([training_process.stdout, training_process.stderr], [], [], 0.1)
                for pipe in rlist:
                    line = pipe.readline().strip()
                    if line:
                        log_queue.put(line)
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {'STDOUT' if pipe == training_process.stdout else 'STDERR'}: {line}\n")
                sys.stdout.flush()
            for line in training_process.stdout:
                if line.strip():
                    log_queue.put(line.strip())
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STDOUT: {line.strip()}\n")
            for line in training_process.stderr:
                if line.strip():
                    log_queue.put(line.strip())
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STDERR: {line.strip()}\n")
            training_process.wait()
            return_code = training_process.returncode
            if return_code == 0 and not process_killed:
                success_msg = f"Training completed, model saved to {model_save_path}"
                log_queue.put(success_msg)
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {success_msg}\n")
            else:
                error_msg = f"Training failed with exit code {return_code}"
                log_queue.put(error_msg)
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {error_msg}\n")
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            log_queue.put(error_msg)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {error_msg}\n")
            sys.stdout.flush()

    if training_process and training_process.poll() is None:
        process_killed = True
        training_process.terminate()
        try:
            training_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            training_process.kill()
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Previous training process terminated\n")
        training_process = None

    training_thread = Thread(target=run_training)
    training_thread.start()
    return jsonify({"message": "训练已开始"})

def stream_logs():
    log_file = os.path.join(BASE_DIR, "train_log.txt")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SSE connection established\n")
    last_heartbeat = time.time()
    while True:
        try:
            log = log_queue.get_nowait()
            yield f"data: {json.dumps({'message': log})}\n\n"
            last_heartbeat = time.time()
        except queue.Empty:
            if process_killed:
                yield f"data: {json.dumps({'message': '训练已停止'})}\n\n"
                break
            # 发送心跳消息防止连接超时
            if time.time() - last_heartbeat > 10:
                yield f"data: {json.dumps({'message': 'Heartbeat: Connection alive'})}\n\n"
                last_heartbeat = time.time()
            time.sleep(0.1)

@app.route('/api/stop', methods=['POST'])
def stop_training():
    global training_process, process_killed
    if training_process and training_process.poll() is None:
        process_killed = True
        training_process.send_signal(signal.SIGTERM)
        try:
            training_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            training_process.kill()
        training_process = None
        log_file = os.path.join(BASE_DIR, "train_log.txt")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] VITS training stopped by user\n")
        # 清空 VITS 日志队列
        while not log_queue.empty():
            log_queue.get()
        return jsonify({"message": "VITS 训练已停止"})
    return jsonify({"error": "没有正在进行的 VITS 训练任务"}), 400

@app.route('/api/train', methods=['GET'])
def stream_training_logs():
    return Response(stream_logs(), mimetype='text/event-stream')

@app.route('/api/outputs/<path:filename>', methods=['GET'])
def download_output():
    try:
        return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 404

@app.route('/api/asr_models', methods=['GET'])
def get_asr_models():
    asr_model_dir = './models/ASR_models'
    try:
        models = [f for f in os.listdir(asr_model_dir) if os.path.isdir(os.path.join(asr_model_dir, f))]
        return jsonify({'models': models})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload_audio', methods=['POST'])
def upload_audio():
    audio_dir = os.path.join(UPLOAD_DIR, 'audio')
    os.makedirs(audio_dir, exist_ok=True)
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    filename = secure_filename(file.filename)
    save_path = os.path.join(audio_dir, filename)
    file.save(save_path)
    return jsonify({'message': '音频上传成功', 'audio_path': save_path})

@app.route('/api/recognize', methods=['POST'])
def recognize_audio():
    data = request.get_json()
    model_name = data.get('model_name')
    audio_path = data.get('audio_path')
    if not model_name or not audio_path:
        return jsonify({'error': '缺少 model_name 或 audio_path'}), 400
    model_dir = os.path.join('./models/ASR_models', model_name)
    test_script = os.path.abspath('test.py')
    conda_activate = 'source ~/anaconda3/bin/activate ASR_train_env'
    cmd = f"{conda_activate} && python {test_script} --model_path '{model_dir}' --audio_path '{audio_path}'"
    try:
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, executable='/bin/bash', timeout=120)
        output = result.stdout + '\n' + result.stderr
        lines = output.splitlines()
        transcription = ''
        for line in lines:
            if line.startswith('Transcription:'):
                transcription = line.replace('Transcription:', '').strip()
                break
        return jsonify({'transcription': transcription, 'raw_output': output})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 只给 ASR 微调用，保存到 Uploads/asr 目录


@app.route('/api/asr_train_logs', methods=['GET'])
def stream_asr_training_logs():
    return Response(stream_asr_logs(), mimetype='text/event-stream')

def stream_logs():
    log_file = os.path.join(BASE_DIR, "train_log.txt")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] VITS SSE connection established\n")
    last_heartbeat = time.time()
    while True:
        try:
            log = log_queue.get_nowait()
            yield f"data: {json.dumps({'message': log})}\n\n"
            last_heartbeat = time.time()
        except queue.Empty:
            if process_killed:
                yield f"data: {json.dumps({'message': 'VITS 训练已停止'})}\n\n"
                break
            if time.time() - last_heartbeat > 10:
                yield f"data: {json.dumps({'message': 'Heartbeat: Connection alive'})}\n\n"
                last_heartbeat = time.time()
            time.sleep(0.1)

def stream_asr_logs():
    log_file = os.path.join(BASE_DIR, "train_log.txt")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ASR SSE connection established\n")
    last_heartbeat = time.time()
    while True:
        try:
            log = asr_log_queue.get_nowait()
            yield f"data: {json.dumps({'message': log})}\n\n"
            last_heartbeat = time.time()
        except queue.Empty:
            if asr_process_killed:
                yield f"data: {json.dumps({'message': 'ASR 训练已停止'})}\n\n"
                break
            if time.time() - last_heartbeat > 10:
                yield f"data: {json.dumps({'message': 'Heartbeat: Connection alive'})}\n\n"
                last_heartbeat = time.time()
            time.sleep(0.1)

@app.route('/api/asr_train', methods=['POST'])
def train_asr_model():
    global asr_training_process, asr_process_killed, asr_training_thread
    data = request.get_json()
    model_name = data.get('model_name')
    dataset_path = data.get('dataset_path')
    training_params = data.get('training_params', {})
    
    if not model_name:
        return jsonify({"error": "缺少 model_name"}), 400
    if not dataset_path:
        return jsonify({"error": "缺少 dataset_path"}), 400
    if not os.path.exists(dataset_path):
        return jsonify({"error": "数据集路径不存在"}), 400

    # 重置 ASR 训练状态
    asr_process_killed = False
    asr_training_process = None
    asr_training_thread = None
    while not asr_log_queue.empty():
        asr_log_queue.get()

    output_dir = os.path.join(OUTPUT_DIR, f"asr_finetuned_{model_name}_{uuid.uuid4().hex[:8]}")
    os.makedirs(output_dir, exist_ok=True)

    conda_env_path = os.path.join(os.path.expanduser("~"), "anaconda3", "envs", "ASR_train_env")
    python_executable = os.path.join(conda_env_path, "bin", "python")
    
    if not os.path.exists(python_executable):
        error_msg = f"Python executable not found: {python_executable}"
        with open(os.path.join(BASE_DIR, "train_log.txt"), 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {error_msg}\n")
        return jsonify({"error": error_msg}), 500

    cmd = [
        python_executable,
        os.path.join(BASE_DIR, "train.py"),
        "--model_path", os.path.join(MODEL_DIR, model_name),
        "--data_dir", dataset_path,
        "--output_dir", output_dir,
        "--batch_size", str(training_params.get('per_device_train_batch_size', 8)),
        "--gradient_accumulation_steps", str(training_params.get('gradient_accumulation_steps', 4)),
        "--num_train_epochs", str(training_params.get('num_train_epochs', 30)),
        "--learning_rate", str(training_params.get('learning_rate', 1e-5)),
        "--save_steps", str(training_params.get('save_steps', 500)),
        "--logging_steps", str(training_params.get('logging_steps', 100)),
        "--save_total_limit", str(training_params.get('save_total_limit', 2)),
        "--fp16", str(training_params.get('fp16', True)).lower()
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PATH"] = f"{os.path.join(conda_env_path, 'bin')}:{env['PATH']}"
    env["LD_LIBRARY_PATH"] = f"{os.path.join(conda_env_path, 'lib')}:/usr/lib/x86_64-linux-gnu:{env.get('LD_LIBRARY_PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    log_file = os.path.join(BASE_DIR, "train_log.txt")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Executing ASR training command: {' '.join(cmd)}\n")
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Environment PATH: {env['PATH']}\n")
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Environment LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}\n")
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Environment CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}\n")
        sys.stdout.flush()

    def run_training():
        global asr_training_process, asr_process_killed
        try:
            test_cmd = [python_executable, "-c", "import sys, torch, librosa, transformers; print('Test: Python version: %s, CUDA: %s' % (sys.version, torch.cuda.is_available())); sys.stdout.flush()"]
            test_process = subprocess.run(
                test_cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test command STDOUT: {test_process.stdout}\n")
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test command STDERR: {test_process.stderr}\n")
                if test_process.returncode != 0:
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test command failed with exit code {test_process.returncode}\n")
            asr_log_queue.put(f"Test command output: {test_process.stdout.strip()}")
            if test_process.stderr:
                asr_log_queue.put(f"Test command error: {test_process.stderr.strip()}")

            asr_training_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=BASE_DIR
            )
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ASR training process started with PID: {asr_training_process.pid}\n")
                sys.stdout.flush()

            from select import select
            while asr_training_process and asr_training_process.poll() is None:
                rlist, _, _ = select([asr_training_process.stdout, asr_training_process.stderr], [], [], 0.1)
                for pipe in rlist:
                    line = pipe.readline().strip()
                    if line:
                        asr_log_queue.put(line)
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {'STDOUT' if pipe == asr_training_process.stdout else 'STDERR'}: {line}\n")
                sys.stdout.flush()
            
            # 检查进程是否已结束
            if asr_training_process:
                for line in asr_training_process.stdout:
                    if line.strip():
                        asr_log_queue.put(line.strip())
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STDOUT: {line.strip()}\n")
                for line in asr_training_process.stderr:
                    if line.strip():
                        asr_log_queue.put(line.strip())
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STDERR: {line.strip()}\n")
                return_code = asr_training_process.wait()
                if return_code == 0 and not asr_process_killed:
                    success_msg = f"Training completed, model saved to {output_dir}"
                    asr_log_queue.put(success_msg)
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {success_msg}\n")
                else:
                    error_msg = f"Training failed with exit code {return_code}"
                    asr_log_queue.put(error_msg)
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {error_msg}\n")
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            asr_log_queue.put(error_msg)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {error_msg}\n")
        finally:
            asr_training_process = None
            asr_training_thread = None
            sys.stdout.flush()

    if asr_training_process and asr_training_process.poll() is None:
        asr_process_killed = True
        asr_training_process.terminate()
        try:
            asr_training_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            asr_training_process.kill()
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Previous ASR training process terminated\n")
        asr_training_process = None
        asr_training_thread = None

    asr_training_thread = Thread(target=run_training)
    asr_training_thread.start()
    return jsonify({"message": "训练已开始", "output_dir": output_dir})

@app.route('/api/asr_stop', methods=['POST'])
def stop_asr_training():
    global asr_training_process, asr_process_killed
    if asr_training_process and asr_training_process.poll() is None:
        asr_process_killed = True
        asr_training_process.send_signal(signal.SIGTERM)
        try:
            asr_training_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            asr_training_process.kill()
        asr_training_process = None
        asr_training_thread = None  # 重置线程
        log_file = os.path.join(BASE_DIR, "train_log.txt")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ASR training stopped by user\n")
        # 清空 ASR 日志队列
        while not asr_log_queue.empty():
            asr_log_queue.get()
        return jsonify({"message": "ASR 训练已停止"})
    return jsonify({"error": "没有正在进行的 ASR 训练任务"}), 400

### 模型测试
@app.route('/api/offline_models', methods=['GET'])
def get_offline_models():
    try:
        models = [f for f in os.listdir(OFFLINE_MODEL_DIR) if os.path.isdir(os.path.join(OFFLINE_MODEL_DIR, f))]
        return jsonify({"models": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/offline_upload_audio', methods=['POST'])
def offline_upload_audio():
    if 'file' not in request.files:
        return jsonify({"error": "没有上传文件"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400
    if file and allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(OFFLINE_UPLOAD_DIR, f"{uuid.uuid4().hex}_{filename}")
        file.save(upload_path)
        return jsonify({"message": "音频上传成功", "audio_path": upload_path})
    return jsonify({"error": "文件格式不支持，仅支持 .wav"}), 400

@app.route('/api/offline_recognize', methods=['POST'])
def offline_recognize():
    data = request.get_json()
    model_name = data.get('model_name')
    audio_path = data.get('audio_path')
    
    if not model_name:
        return jsonify({"error": "缺少 model_name"}), 400
    if not audio_path:
        return jsonify({"error": "缺少 audio_path"}), 400
    if not os.path.exists(audio_path):
        return jsonify({"error": "音频文件不存在"}), 400
    if not os.path.exists(os.path.join(OFFLINE_MODEL_DIR, model_name)):
        return jsonify({"error": "模型路径不存在"}), 400

    conda_env_path = os.path.join(os.path.expanduser("~"), "anaconda3", "envs", "ASR_train_env")
    python_executable = os.path.join(conda_env_path, "bin", "python")
    
    if not os.path.exists(python_executable):
        return jsonify({"error": f"Python executable not found: {python_executable}"}), 500

    cmd = [
        python_executable,
        os.path.join(BASE_DIR, "test.py"),
        "--model_path", os.path.join(OFFLINE_MODEL_DIR, model_name),
        "--audio_path", audio_path
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PATH"] = f"{os.path.join(conda_env_path, 'bin')}:{env['PATH']}"
    env["LD_LIBRARY_PATH"] = f"{os.path.join(conda_env_path, 'lib')}:/usr/lib/x86_64-linux-gnu:{env.get('LD_LIBRARY_PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    log_file = os.path.join(BASE_DIR, "offline_recognition_log.txt")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Executing recognition command: {' '.join(cmd)}\n")
        sys.stdout.flush()

    try:
        process = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60
        )
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STDOUT: {process.stdout}\n")
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STDERR: {process.stderr}\n")
            sys.stdout.flush()

        if process.returncode == 0:
            transcription = process.stdout.strip().split("Transcription:")[-1].strip() if "Transcription:" in process.stdout else ""
            return jsonify({"transcription": transcription, "raw_output": process.stdout})
        else:
            error_msg = f"Recognition failed with exit code {process.returncode}: {process.stderr}"
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {error_msg}\n")
            return jsonify({"error": error_msg, "raw_output": process.stderr}), 500
    except subprocess.TimeoutExpired:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Recognition timed out\n")
        return jsonify({"error": "Recognition timed out"}), 500
    except Exception as e:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Recognition failed: {str(e)}\n")
        return jsonify({"error": f"Recognition failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)