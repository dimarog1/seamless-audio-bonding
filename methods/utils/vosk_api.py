import json
import wave

import librosa
import soundfile
from vosk import KaldiRecognizer, Model


class RecognizerWrapper:
    chunk_size = 4000
    sample_rate = 22050

    def __init__(self, model_directory: str, sr: int = 22050):
        self.model_directory = model_directory
        self.temporary_audio_path = 'vosk_data/tmp/tmp.wav'  # путь к временному файлу vosk
        self.model = Model(model_directory)
        self.sample_rate = sr

    def _process_sample(self):
        recognizer = KaldiRecognizer(self.model, self.sample_rate)
        recognizer.SetWords(True)

        audio = wave.open(self.temporary_audio_path, 'rb')

        results = []
        while True:
            data = audio.readframes(self.chunk_size)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                part_result = json.loads(recognizer.Result())
                results.append(part_result)

        part_result = json.loads(recognizer.FinalResult())
        results.append(part_result)

        return results

    def get_transcription(self, audio_path: str):
        detected_words, time_steps = [], []

        audio_sample, _ = librosa.load(audio_path, sr=self.sample_rate)
        soundfile.write(
            file=self.temporary_audio_path,
            data=audio_sample,
            samplerate=self.sample_rate,
        )

        results = self._process_sample()

        for utterance_result in results:
            if 'result' not in utterance_result:
                continue
            for word_result in utterance_result['result']:
                if not len(word_result['word']):
                    continue

                word = word_result['word']
                start = word_result['start']
                end = word_result['end']

                detected_words.append(word)
                time_steps.append((start, end))

        return {
            "words": detected_words,
            "time_steps": time_steps,
        }