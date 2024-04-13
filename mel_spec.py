import numpy as np
from tqdm import tqdm
from methods.utils.meldataset import mel_spectrogram
import argparse
import torchaudio
import os
import warnings

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--wavs_dir', default=None)
    parser.add_argument('--save_dir', default=None)

    a = parser.parse_args()

    for number, file in tqdm(enumerate(os.listdir(a.wavs_dir))):
        audio, _ = torchaudio.load(a.wavs_dir + f'/{file}')
        mel_splice = mel_spectrogram(audio, 1024, 80, 22050, 256, 1024, 0, 8000, center=False)

        np.save(a.save_dir + f'/orig{number}.npy', mel_splice)

    print('Done')


if __name__ == '__main__':
    main()
