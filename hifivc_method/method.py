import torch
import os
from .audio_utils import merge_audio, preprocess_audio, mel_spectrogram, smooth_pitch, fade_pieces, AudioFeaturesParams, SAMPLE_RATE
from .f0_utils import get_lf0_from_wav
import numpy as np
import soundfile as sf


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = 'cuda'
params = AudioFeaturesParams()


def get_files(path: str):
    return sorted([f'{path}/{file}' for file in os.listdir(path)])


def smooth_audio(audio, *samples):
    for sample in samples:
        fade_duration = int(0.12 * SAMPLE_RATE)
        duration = fade_duration // 2
        fade_pieces(audio[:sample], audio[sample:], duration, duration, 1, 0.054, 1)
    return audio


def save_audio(audio: np.array, path: str, sr=22050) -> None:
    sf.write(f'{path}/result.wav', audio, sr)


def fade_method(audios_path: str, save_path: str):
    merged, splices = merge_audio(*get_files(audios_path))
    smoothed = smooth_audio(merged, *splices)

    save_audio(smoothed, save_path, sr=SAMPLE_RATE)


def convert_method(audios_path: str, save_path: str, model_jit_path: str):
    merged, _ = merge_audio(*get_files(audios_path))

    sf.write(f'tmp.wav', merged, SAMPLE_RATE)

    wav_source = preprocess_audio(merged).to(device)

    wav_ref = preprocess_audio(merged)
    mel_ref = mel_spectrogram(wav_ref, params).to(device)
    pitch = get_lf0_from_wav('tmp.wav').to(device).float()
    
    with torch.no_grad():
        traced = torch.jit.load(model_jit_path).eval()
        converted = traced(wav_source, mel_ref, pitch)
    
    save_audio(converted.cpu().squeeze().detach().numpy(), save_path)


def fade_convert_method(audios_path: str, save_path: str, model_jit_path: str):
    merged, splices = merge_audio(*get_files(audios_path))
    smoothed = smooth_audio(merged, *splices)

    sf.write(f'tmp.wav', merged, SAMPLE_RATE)

    wav_source = preprocess_audio(merged).to(device)

    wav_ref = preprocess_audio(merged)
    mel_ref = mel_spectrogram(wav_ref, params).to(device)
    pitch = get_lf0_from_wav('tmp.wav').to(device).float()
    
    with torch.no_grad():
        traced = torch.jit.load(model_jit_path).eval()
        converted = traced(wav_source, mel_ref, pitch)
    
    save_audio(converted.cpu().squeeze().detach().numpy(), save_path)


def smooth_pitch_convert_method(audios_path: str, save_path: str, model_jit_path: str):
    merged, splices = merge_audio(*get_files(audios_path))
    
    sf.write(f'tmp.wav', merged, SAMPLE_RATE)

    wav_source = preprocess_audio(merged).to(device)

    wav_ref = preprocess_audio(merged)
    mel_ref = mel_spectrogram(wav_ref, params).to(device)
    pitch = get_lf0_from_wav('tmp.wav').to(device).float()
    
    smooth_pitch(merged, pitch, *splices)
    
    with torch.no_grad():
        traced = torch.jit.load(model_jit_path).eval()
        converted = traced(wav_source, mel_ref, pitch)
    
    save_audio(converted.cpu().squeeze().detach().numpy(), save_path)
