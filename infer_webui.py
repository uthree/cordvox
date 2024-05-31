import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchaudio.functional import resample

import lightning as L

from module.cordvox import Cordvox
from module.utils.config import load_json_file
from module.utils.f0_estimation import estimate_f0
from module.utils.mel import LogMelSpectrogram

import gradio as gr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config/v3_24k.json")
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("-p", "--pitch-shift", default=0.0, type=float)
    args = parser.parse_args()

    config = load_json_file(args.config)
    model_path = Path(config['save']['models_dir']) / "model.ckpt"
    device = torch.device(args.device)

    print(f"loading model from {model_path}")
    model = Cordvox.load_from_checkpoint(model_path, map_location=device)

    generator = model.generator.to(device)
    sample_rate = 24000

    frame_size = config["preprocess"]["frame_size"]
    length = config['preprocess']['length']
    sample_rate = config["preprocess"]["sample_rate"]
    pe_algorithm = config['preprocess']['pitch_estimation']
    n_mels = config['preprocess']['n_mels']
    n_fft = config['preprocess']['n_fft']

    to_mel = LogMelSpectrogram(sample_rate, n_fft, frame_size, n_mels).to(device)

    @torch.inference_mode()
    def convert(input_audio):
        input_sr, input_wf = input_audio
        input_wf = torch.from_numpy(input_wf).unsqueeze(0).to(device).to(torch.float) / 32768.0
        input_wf = resample(input_wf, input_sr, sample_rate).to(device)

        f0 = estimate_f0(input_wf, sample_rate, frame_size, pe_algorithm)
        mel = to_mel(input_wf)
        output_wf = generator(mel, f0).squeeze(1)
       
        output_wf = output_wf.clamp(-1.0, 1.0)
        output_wf = output_wf * 32768.0
        output_wf = output_wf.to(torch.int16).squeeze(0).cpu().numpy()
        return (sample_rate, output_wf)
    
    demo = gr.Interface(
        convert,
        inputs=[
            gr.Audio(label="Input"),
        ],
        outputs=[
            gr.Audio(label="Output")
        ]
    )
    
    demo.launch()