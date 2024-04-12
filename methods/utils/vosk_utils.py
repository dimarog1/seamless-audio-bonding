from typing import List

from .vosk_api import *

SAMPLE_RATE = 22050


def determine_words(audio_path: str, vosk_model: str, sr: int = SAMPLE_RATE):
    try:
        rec_wrapper = RecognizerWrapper(vosk_model + '/model/vosk-model-en-us-0.22-lgraph' , sr)
    except Exception as e:
        raise RuntimeError('Incorrect vosk model path')
    result = rec_wrapper.get_transcription(audio_path)
    zipped_recognition = list(zip(result['words'], result['time_steps']))

    return zipped_recognition


def get_neighbors(data, splice_samples: List[int], vosk_model: str, source_path='tmp.wav'):
    global SAMPLE_RATE
    soundfile.write(source_path, data, SAMPLE_RATE)
    data, sr = librosa.load(source_path)
    SAMPLE_RATE = sr
    ascending_recognition = determine_words(source_path, vosk_model)

    error = 5  # %
    res_list = []

    for sample in splice_samples:
        for i in range(len(ascending_recognition)):
            word, span = ascending_recognition[i]
            l, r = span
            if abs(1 - l * SAMPLE_RATE / sample) * 100 <= error:
                res_list.append((int(ascending_recognition[i - 1 if i - 1 >= 0 else 0][1][0] * SAMPLE_RATE),
                                 int(r * SAMPLE_RATE)))
                break
        else:
            res_list.append((None, None))

    return res_list