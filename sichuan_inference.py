import os
import torch
import argparse
from pypinyin import pinyin, Style
import commons
import utils
from models import SynthesizerTrn
from text import text_to_sequence
import soundfile as sf

# ...existing code...
def get_text(text, hps):
    print(f"Input text: {text}")
    # Hardcode text_cleaners to avoid any external modification
    text_cleaners = ['chinese_cleaners']
    print(f"Text cleaners: {text_cleaners}")
    print(f"Use pinyin: {hps.use_pinyin}")
    # Convert to pinyin if enabled
    if hps.use_pinyin:
        text = " ".join([item[0] for item in pinyin(text, style=Style.TONE3)])
        print(f"Pinyin text: {text}")
    # Pass cleaners as the third argument and symbols as the second
    text_norm = text_to_sequence(text, hps.symbols, text_cleaners)
    print(f"Text norm: {text_norm}")
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
        print(f"Text norm with blanks: {text_norm}")
    return torch.LongTensor(text_norm)
# ...existing code...

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help='Path to model checkpoint (G_XXXXX.pth)')
    parser.add_argument('-c', '--config', required=True, help='Path to config file')
    parser.add_argument('-t', '--text', required=True, help='Text to synthesize')
    parser.add_argument('-s', '--speaker', default='0', help='Speaker ID')
    parser.add_argument('-l', '--language', default='Sichuanese', help='Language')
    parser.add_argument('-ls', '--length_scale', type=float, default=1.0, help='Length scale')
    parser.add_argument('--noise_scale', type=float, default=0.3, help='Noise scale')
    parser.add_argument('--noise_scale_w', type=float, default=0.5, help='Noise scale for duration')
    parser.add_argument('--use_pinyin', action='store_true', help='Use pinyin input')
    args = parser.parse_args()

    hps = utils.get_hparams_from_file(args.config)
    hps.use_pinyin = args.use_pinyin
    print(f"Loaded config: {args.config}")
    print(f"hps.data.text_cleaners: {hps.data.text_cleaners}")

    net_g = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda()
    net_g.eval()
    utils.load_checkpoint(args.model, net_g, None)

    text_norm = get_text(args.text, hps).cuda()
    text_norm = text_norm[None]
    text_lengths = torch.LongTensor([text_norm.size(1)]).cuda()
    speaker_id = torch.LongTensor([int(args.speaker)]).cuda()

    with torch.no_grad():
        audio = net_g.infer(
            text_norm, text_lengths, speaker_id,
            noise_scale=args.noise_scale,
            noise_scale_w=args.noise_scale_w,
            length_scale=args.length_scale
        )[0][0, 0].data.cpu().float().numpy()
    os.makedirs('output', exist_ok=True)
    output_path = f'output/sichuan_{args.speaker}_{args.language}.wav'
    sf.write(output_path, audio, hps.data.sampling_rate)
    print(f"Generated audio saved to {output_path}")

if __name__ == "__main__":
    main()