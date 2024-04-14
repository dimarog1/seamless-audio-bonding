from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import json
import torch
from scipy.io.wavfile import write
from .utils.env import AttrDict
from .utils.meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from .utils.models import Generator
import warnings

warnings.filterwarnings("ignore")

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    checkpoint_dict = torch.load(filepath, map_location=device)
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(audios_path: str, save_path: str, hifigan_model_path: str):
    generator = Generator(h).to(device)

    try:
        state_dict_g = load_checkpoint(hifigan_model_path, device)
    except Exception:
        raise RuntimeError('Incorrect hifi-gan model path')
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(audios_path)

    os.makedirs(save_path, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            wav, sr = load_wav(os.path.join(audios_path, filname))
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            x = get_mel(wav.unsqueeze(0))
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(save_path, 'result.wav')
            write(output_file, 16000, audio)


def hifigan(audios_path: str, save_path: str, hifigan_model_path: str, config_file: str):
    try:
        with open(config_file) as f:
            data = f.read()

        global h
        json_config = json.loads(data)
        h = AttrDict(json_config)

        torch.manual_seed(h.seed)
        global device
        if torch.cuda.is_available():
            torch.cuda.manual_seed(h.seed)
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    except Exception:
        raise RuntimeError('Incorrect config file path')

    inference(audios_path, save_path, hifigan_model_path)
