import torch
import os
from .utils.audio_utils import merge_audio, preprocess_audio, mel_spectrogram, smooth_pitch, fade, AudioFeaturesParams, SAMPLE_RATE, get_files, save_audio
from .utils.f0_utils import get_lf0_from_wav
import numpy as np
import soundfile as sf


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = 'cuda'
params = AudioFeaturesParams()


def smooth_audio(audio: np.array, *samples: int):
    for sample in samples:
        fade_duration = int(0.12 * SAMPLE_RATE)
        duration = fade_duration // 2
        fade(audio[:sample], audio[sample:], duration, duration, 1, 0.054, 1)
    return audio


def fade_method(audios_path: str, save_path: str):
    merged, splices, _ = merge_audio(*get_files(audios_path))
    smoothed = smooth_audio(merged, *splices)

    save_audio(smoothed, save_path, sr=SAMPLE_RATE)


def convert_method(audios_path: str, save_path: str, model_jit_path: str):
    merged, _, _ = merge_audio(*get_files(audios_path))

    sf.write('tmp.wav', merged, SAMPLE_RATE)

    wav_source = preprocess_audio(merged).to(device)

    wav_ref = preprocess_audio(merged)
    mel_ref = mel_spectrogram(wav_ref, params).to(device)
    pitch = get_lf0_from_wav('tmp.wav').to(device).float()
    
    with torch.no_grad():
        traced = torch.jit.load(model_jit_path).eval()
        converted = traced(wav_source, mel_ref, pitch)
    
    save_audio(converted.cpu().squeeze().detach().numpy(), save_path)


def fade_convert_method(audios_path: str, save_path: str, model_jit_path: str):
    merged, splices, _ = merge_audio(*get_files(audios_path))
    smoothed = smooth_audio(merged, *splices)

    sf.write('tmp.wav', smoothed, SAMPLE_RATE)

    wav_source = preprocess_audio(merged).to(device)

    wav_ref = preprocess_audio(merged)
    mel_ref = mel_spectrogram(wav_ref, params).to(device)
    pitch = get_lf0_from_wav('tmp.wav').to(device).float()
    
    with torch.no_grad():
        traced = torch.jit.load(model_jit_path).eval()
        converted = traced(wav_source, mel_ref, pitch)
    
    save_audio(converted.cpu().squeeze().detach().numpy(), save_path)


def smooth_pitch_convert_method(audios_path: str, save_path: str, model_jit_path: str):
    merged, splices, _ = merge_audio(*get_files(audios_path))
    
    sf.write('tmp.wav', merged, SAMPLE_RATE)

    wav_source = preprocess_audio(merged).to(device)

    wav_ref = preprocess_audio(merged)
    mel_ref = mel_spectrogram(wav_ref, params).to(device)
    pitch = get_lf0_from_wav('tmp.wav').to(device).float()
    
    smooth_pitch(merged, pitch, *splices)
    
    with torch.no_grad():
        traced = torch.jit.load(model_jit_path).eval()
        converted = traced(wav_source, mel_ref, pitch)
    
    save_audio(converted.cpu().squeeze().detach().numpy(), save_path)
