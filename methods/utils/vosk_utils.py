from typing import List

from .vosk_api import *


def determine_words(audio_path: str, sr: int):
    rec_wrapper = RecognizerWrapper(MODEL_PATH, sr)
    result = rec_wrapper.get_transcription(audio_path)
    zipped_recognition = list(zip(result['words'], result['time_steps']))

    return zipped_recognition


def get_neighbors(source_path: str, splices: List[int]):
    data, sr = librosa.load(source_path)

    # TODO: neighbors
    pass