import torch
import os
import librosa
from audio_utils import merge_audio, preprocess_audio, mel_spectrogram, smooth_pitch, fade_pieces
from f0_utils import get_lf0_from_wav
import numpy as np


device = 'cuda'
SAMPLE_RATE = 22050


def smooth_audio(audio, *samples):
    for ind, sample in enumerate(samples):
        if 0 < ind < len(samples) - 1:
            fade_duration = min(len(audio[samples[ind - 1]:sample]), len(audio[sample:samples[ind + 1]])) // 2
        elif ind == 0 and len(samples) > 1:
            fade_duration = min(len(audio[:sample]), len(audio[sample:samples[ind + 1]])) // 2
        elif ind == len(samples) - 1 and len(samples) > 1:
            fade_duration = min(len(audio[samples[ind - 1]:sample]), len(audio[sample:])) // 2
        duration = fade_duration // 2
        fade_pieces(audio[:sample], audio[sample:], duration, duration, 1, 0.1, 1)
    return audio


def save_audio(audio: np.array, path: str) -> None:
    librosa.output.write_wav(f'{path}/result.wav', audio, SAMPLE_RATE)


def fade_method(audios_path: str, save_path: str):
    merged, splices = merge_audio(*os.listdir(audios_path))
    smoothed = smooth_audio(merged, *splices)

    save_audio(smoothed, save_path)


def convert_method(audios_path: str, save_path: str, model_jit_path: str):
    merged, _ = merge_audio(*os.listdir(audios_path))

    save_audio(merged, 'tmp.wav')

    wav_source = preprocess_audio(merged).to(device)

    wav_ref = preprocess_audio(merged)
    mel_ref = mel_spectrogram(wav_ref).to(device)
    pitch = get_lf0_from_wav('tmp.wav')
    
    with torch.no_grad():
        traced = torch.jit.load(model_jit_path).eval()
        converted = traced(wav_source, mel_ref, pitch)
    
    save_audio(converted, save_path)


def fade_convert_method(audios_path: str, save_path: str, model_jit_path: str):
    merged, splices = merge_audio(*os.listdir(audios_path))
    smoothed = smooth_audio(merged, *splices)

    save_audio(smoothed, 'tmp.wav')

    wav_source = preprocess_audio(merged).to(device)

    wav_ref = preprocess_audio(merged)
    mel_ref = mel_spectrogram(wav_ref).to(device)
    pitch = get_lf0_from_wav('tmp.wav')
    
    with torch.no_grad():
        traced = torch.jit.load(model_jit_path).eval()
        converted = traced(wav_source, mel_ref, pitch)
    
    save_audio(converted, save_path)


def smooth_pitch_convert_method(audios_path: str, save_path: str, model_jit_path: str):
    merged, splices = merge_audio(*os.listdir(audios_path))
    
    save_audio(merged, 'tmp.wav')

    wav_source = preprocess_audio(merged).to(device)

    wav_ref = preprocess_audio(merged)
    mel_ref = mel_spectrogram(wav_ref).to(device)
    pitch = get_lf0_from_wav('tmp.wav')
    
    smooth_pitch(merged, pitch, *splices)
    
    with torch.no_grad():
        traced = torch.jit.load(model_jit_path).eval()
        converted = traced(wav_source, mel_ref, pitch)
    
    save_audio(converted, save_path)
