# SeamlessAudioBonding: приложения для бесшовной склейки аудио

## Описание

Приложение поддерживает множество методов склейки аудио, а именно:

* С помощью линейных и экспоненциальных фейдов с фиксированной длиной и в зависимости от длины наименьшего из ближайших слов.
* С помощью технологии [HIFI-VC](https://github.com/tinkoff-ai/hifi_vc.git).
* С помощью технологии [HIFI-GAN](https://github.com/jik876/hifi-gan.git).

## Установка

> Компьютер должен обязательно поддерживать технологию cuda!

### Установка приложения

Можно скачать архив с [репозитория](https://github.com/dimarog1/seamless-audio-bonding.git).

Или же установить с помощью git:

``` shell
git clone https://github.com/dimarog1/seamless-audio-bonding.git
```

### Установка зависимостей (Wheel)

Предварительные условия:

* Python >= 3.10

Для установки зависимостей пользуйтесь файлом [requirements.txt](./requirements.txt):

``` shell
pip install -r ./requirements.txt
```

Для установки pytorch отдельно воспользуйтесь методами установки с [официального сайта](https://pytorch.org/get-started/previous-versions/):

``` shell
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

Послу установки PyTorch может слететь версия librosa, она должна быть версии **0.9.1**:

``` shell
pip install librosa==0.9.1
```

### Установка моделей

Модель [Vosk](https://alphacephei.com/vosk/models), в нашем приложении используется модель [vosk-model-en-us-0.22-lgraph](https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip).

Модель [HIFI-VC](https://drive.google.com/file/d/1oFwMeuQtwaBEyOFkyG7c7LfBQiRe3RdW/view).

Модель [HIFI-GAN](https://disk.yandex.ru/d/PWkFYPZL5pGBKA), для склейки аудио хватит только генератора (g_...), для обучения потребуются все файлы.

## Пользование приложением

### Склейка аудио

Для склейки аудио используется скрипт [mix.py](./mix.py), для него предусмотрены следующие флаги:

* --method - используется для выбора метода склеивания аудиофайлов, могут быть выбраны следующие методы:
  * linear_time_fade - линейный фейд с фиксированной длиной перехода.
  * linear_word_fade - линейный фейд с длиной перехода, зависящей от ближайшего минимального слова.
  * exp_time_fade - экспоненциальный фейд с фиксированной длиной перехода.
  * exp_word_fade - экспоненциальный фейд с длиной перехода, зависящей от ближайшего минимального слова.
  * convert - преобразование в тот же самый голос с помощью технологии [HIFI-VC](https://github.com/tinkoff-ai/hifi_vc.git).
  * fade_convert - линейный фейд с фиксированной длиной перехода, а затем преобразование в тот же самый голос с помощью технологии [HIFI-VC](https://github.com/tinkoff-ai/hifi_vc.git).
  * smooth_pitch - преобразование в тот же самый голос с помощью технологии [HIFI-VC](https://github.com/tinkoff-ai/hifi_vc.git) с редактированием интонации.
  * hifigan - склеивание аудиофайлов с помощью дообученной модели технологии [HIFI-GAN](https://github.com/jik876/hifi-gan.git).
* --audios_path - директория, где лежат аудиофайлы, которые нужно склеить, причём они должны быть пронумерованы в соответствии с их порядком.
* --save_path - директория, куда требуется сохранить склеенное аудио.
* --hifivc_model_path - путь до модели [HIFI-VC](https://drive.google.com/file/d/1oFwMeuQtwaBEyOFkyG7c7LfBQiRe3RdW/view).
* --vosk_model_path - путь до модели [Vosk](https://alphacephei.com/vosk/models).
* --hifigan_model_path - путь до модели [HIFI-GAN](https://disk.yandex.ru/d/PWkFYPZL5pGBKA).

Пример:

``` shell
python mix.py --method exp_time_fade --audios_path audios --save_path smoothed_audio --hifivc_model_path model.pt --vosk_model_path vosk_data --hifigan_model_path cp_hifigan/g_02700000
```

Также некоторые параметры установлены по умолчанию, это можно посмотреть в самом [скрипте](./mix.py).

### Обучение модели

Для обучения модели на основе технологии [HIFI-GAN](https://github.com/jik876/hifi-gan.git) используется скрипт [train.py](./train.py), для него предусмотрены следующие флаги:

* --input_wavs_dir - директория с оригинальными (исходными) аудиофайлами.
* --input_mels_dir - директория с mel-спектрограммами для обучения с параметром fine-tuning.
* --input_training_file - текстовый файл, где на каждой строке написан файл, который будет использоваться для обучения модели, пишется без расширения.
* --input_validation_file - текстовый файл, где на каждой строке написан файл, который будет использоваться для валидации модели, пишется без расширения.
* --checkpoint_path - директория, куда будут сохраняться логи и чекпоинты при обучени модели.
* --config - путь до конфигурационного файла.
* --training_epochs - количество эпох обучения.
* --stdout_interval - интервал (шагов) вывода небольшого отчёта о процессе обучения модели.
* --checkpoint_interval - интервал создания чекпоинта модели.
* --summary_interval - интервал логирования процесса обучения модели.
* --validation_interval - интервал валидации модели.
* --fine_tuning - настройка, которой устанавливается значение True, когда для обучения используются собственные mel-спектрограммы.
* --training_data_percent - процент, который используется для обучения модели, 100 - training_data_percent процентов датасета будет использоваться для валидации модели.

Пример:

``` shell
python train.py --input_wavs_dir data/validation/origs --input_mels_dir mel_files/spliced --input_training_file data/training.txt --input_validation_file data/validation.txt --checkpoint_path cp_hifigan --config configs/config.json --training_epochs 3100 --stdout_interval 5 --checkpoint_interval 5000 --summary_interval 100 --validation_interval 1000 --fine_tuning True --training_data_percent 80
```

Также некоторые параметры установлены по умолчанию, это можно посмотреть в самом [скрипте](./train.py).

### Преобразование аудиофайлов в mel-спектрограммы

Для преобразование аудиофайлов в mel-спектрограммы используется скрипт [mel_spec.py](./mel_spec.py), для него предусмотрены следующие флаги:

* --wavs_dir - директория с аудиофайлами.
* --save_dir - директория, куда требуется сохранить полученные mel-спектрограммы.

Пример:

``` shell
python mel_spec.py --wavs_dir data/training/origs --save_dir mel_files/spliced
```
