import argparse
from hifivc_method.method import fade_method, convert_method, fade_convert_method, smooth_pitch_convert_method


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--method', default='fade', type=str)
    parser.add_argument('--audios_path', default='audios', type=str)
    parser.add_argument('--save_path', default='smoothed_audio', type=str)
    parser.add_argument('--model_path', default='model.pt', type=str)
    
    a = parser.parse_args()
    
    if a.method == 'fade':
        fade_method(a.audios_path, a.save_path)
    elif a.method == 'convert':
        convert_method(a.audios_path, a.save_path, a.model_path)
    elif a.method == 'fade_convert':
        fade_convert_method(a.audios_path, a.save_path, a.model_path)
    elif a.method == 'smooth_pitch':
        smooth_pitch_convert_method(a.audios_path, a.save_path, a.model_path)


if __name__ == '__main__':
    main()
