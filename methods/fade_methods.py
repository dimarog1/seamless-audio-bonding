import numpy as np
from .utils.audio_utils import fade, merge_audio, get_files, save_audio


# TODO VOSK - сделать функцию, которая будет давать координаты ближайших справа и слева слов от места склейки в аудио.
# Линейный фэйд с оптимальной длиной минимального ближайшего слова
def linear_word(audios_path: str, save_path: str, center_fade=0.054, fade_len=3.542):
    audio, splices, sr = merge_audio(*get_files(audios_path))
    
    for ind, sample in enumerate(splices):
        if 0 < ind < len(splices) - 1:
            fade_duration = int(
                min(len(audio[splices[ind - 1]:sample]), len(audio[sample:splices[ind + 1]])) / fade_len)
        elif ind == 0 and len(splices) > 1:
            fade_duration = int(min(len(audio[:sample]), len(audio[sample:splices[ind + 1]])) / fade_len)
        elif ind == len(splices) - 1 and len(splices) > 1:
            fade_duration = int(min(len(audio[splices[ind - 1]:sample]), len(audio[sample:])) / fade_len)
        duration = int(fade_duration // 2)
        fade(audio[:sample], audio[sample:], duration, duration, 1.0, center_fade, 1.0)
    
    save_audio(audio, save_path, sr)


# Линейный фэйд с оптимальной длиной в секундах
def linear_time(audios_path: str, save_path: str, center_fade=0.033, fade_duration=0.12):
    audio, splices, sr = merge_audio(*get_files(audios_path))

    duration = int((fade_duration * sr) // 2)
    for sample in splices:
        fade(audio[:sample], audio[sample:], duration, duration, 1.0, center_fade, 1.0)
    
    save_audio(audio, save_path, sr)


# TODO VOSK - сделать функцию, которая будет давать координаты ближайших справа и слева слов от места склейки в аудио.
# Экспоненциальный фэйд с оптимальной длиной минимального ближайшего слова и силой фейда
def exp_word(audios_path: str, save_path: str, center_fade=0.01, fade_len=3.93, fade_power=1.09):
    audio, splices, sr = merge_audio(*get_files(audios_path))

    for ind, sample in enumerate(splices):
        if 0 < ind < len(splices) - 1:
            fade_duration = int(
                min(len(audio[splices[ind - 1]:sample]), len(audio[sample:splices[ind + 1]])) / fade_len)
        elif ind == 0 and len(splices) > 1:
            fade_duration = int(min(len(audio[:sample]), len(audio[sample:splices[ind + 1]])) / fade_len)
        elif ind == len(splices) - 1 and len(splices) > 1:
            fade_duration = int(min(len(audio[splices[ind - 1]:sample]), len(audio[sample:])) / fade_len)
        duration = int(fade_duration // 2)
        fade(audio[:sample], audio[sample:], duration, duration, 1.0, center_fade, 1.0, exp=fade_power)
    
    save_audio(audio, save_path, sr)


# Экспоненциальный фэйд с оптимальной длиной в секундах и силой фейда
def exp_time(audios_path: str, save_path: str, center_fade=0.012, fade_duration=0.072, fade_power=1.66):
    audio, splices, sr = merge_audio(*get_files(audios_path))

    duration = int((fade_duration * sr) // 2)
    for sample in splices:
        fade(audio[:sample], audio[sample:], duration, duration, 1.0, center_fade, 1.0, exp=fade_power)
    
    save_audio(audio, save_path, sr)
