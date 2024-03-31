import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--method', default='fade', type=str)
    parser.add_argument('--audios_path', default='audios', type=str)
    parser.add_argument('--save_path', default='smoothed_audio', type=str)
    parser.add_argument('--model_path', default='models/model.pt', type=str)


if __name__ == '__main__':
    pass