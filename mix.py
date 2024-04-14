import argparse
from methods.hifivc_method import convert_method, fade_convert_method, smooth_pitch_convert_method
from methods.fade_methods import linear_word, linear_time, exp_word, exp_time
from methods.hifigan_method import hifigan

import os.path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--method', default='exp_time_fade', type=str)
    parser.add_argument('--audios_path', default='audios', type=str)
    parser.add_argument('--save_path', default='smoothed_audio', type=str)
    parser.add_argument('--hifivc_model_path', default='model.pt', type=str)
    parser.add_argument('--vosk_model_path', default='vosk_data', type=str)
    parser.add_argument('--hifigan_model_path', default='cp_hifigan/g_02700000', type=str)
    parser.add_argument('--config', default='configs/config.json', type=str)
    
    a = parser.parse_args()
    
    if not os.path.isdir(a.audios_path):
        raise FileNotFoundError('Incorrect input audio files')
    if not os.path.isdir(a.save_path):
        raise FileNotFoundError('Incorrect output path')
    
    if a.method == 'convert':
        convert_method(a.audios_path, a.save_path, a.hifivc_model_path)
    elif a.method == 'fade_convert':
        fade_convert_method(a.audios_path, a.save_path, a.hifivc_model_path)
    elif a.method == 'smooth_pitch':
        smooth_pitch_convert_method(a.audios_path, a.save_path, a.hifivc_model_path)
    elif a.method == 'linear_word_fade':
        linear_word(a.audios_path, a.save_path, a.vosk_model_path)
    elif a.method == 'linear_time_fade':
        linear_time(a.audios_path, a.save_path)
    elif a.method == 'exp_word_fade':
        exp_word(a.audios_path, a.save_path, a.vosk_model_path)
    elif a.method == 'exp_time_fade':
        exp_time(a.audios_path, a.save_path)
    elif a.method == 'hifigan':
        exp_time(a.audios_path, a.save_path)
        hifigan(a.save_path, a.save_path, a.hifigan_model_path, a.config)


if __name__ == '__main__':
    try:
        main()
        print('Done')
    except Exception as error:
        print(error)
