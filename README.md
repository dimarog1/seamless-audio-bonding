# SeamlessAudioBonding: приложения для бесшовной склейки аудио

## Описание

Приложение поддерживает множество методов склейки аудио, а именно:

* С помощью линейных и экспоненциальных фейдов с фиксированной длиной и в зависимости от длины наименьшего из ближайших слов.
* С помощью технологии [HIFI-VC](https://github.com/tinkoff-ai/hifi_vc.git).
* С помощью технологии [HIFI-GAN](https://github.com/jik876/hifi-gan.git).

## Установка

> Видеокарта должна обязательно поддерживать архитектуру **CUDA**!

### Установка приложения

Можно скачать архив с [репозитория](https://github.com/dimarog1/seamless-audio-bonding.git).

Или же установить с помощью git:

```shell
git clone https://github.com/dimarog1/seamless-audio-bonding.git
```

### Установка зависимостей (Wheel)

Обязательные условия:

* Python 3.10

Для установки зависимостей пользуйтесь файлом [requirements.txt](./requirements.txt):

```shell
pip install -r ./requirements.txt
```

Для установки pytorch отдельно воспользуйтесь методами установки с [официального сайта](https://pytorch.org/get-started/previous-versions/):

```shell
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

После установки всех зависимостей может слететь версия librosa, она должна быть версии **0.9.1**:

```shell
pip install librosa==0.9.1
```

### Установка моделей

В нашем приложении используется [vosk-model-en-us-0.22-lgraph](https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip), однако использовать можно любую с официального сайта [VOSK-API](https://alphacephei.com/vosk/models);

Модель [HIFI-VC](https://drive.google.com/file/d/1oFwMeuQtwaBEyOFkyG7c7LfBQiRe3RdW/view);

Модель [HIFI-GAN](https://disk.yandex.ru/d/PWkFYPZL5pGBKA), для склейки аудио хватит только генератора (файл с названием g_...), для обучения потребуются все файлы. Желательно, чтобы генератор и дискриминатор (файл с названием do_...) находились в директории с чекпоинтами модели (например, cp_hifigan).

### Установка VOSK-API

Все необходимые зависимости уже находятся в [requirements.txt](./requirements.txt).

Путь к директории VOSK присваивается переменной VOSK_DATA в скрипте [vosk_api.py](./methods/utils/vosk_api.py). Для корректной работы необходимо, чтобы директория VOSK_DATA содержала 2 папки: model и tmp. 

* model должна содержать сам файл с моделью (по умолчанию ожидается vosk-model-en-us-0.22-lgraph). При желании изменить целевую модель, необходимо вручную изменить переменную MODEL в скрипте [vosk_api.py](./methods/utils/vosk_api.py));
* tmp должна содержать временный аудиофайл tmp.wav (он может быть пустым).

## Использование приложения

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
* --config - путь до конфигурационного файла (используется при применении технологии [HIFI-GAN](https://github.com/jik876/hifi-gan.git)).

Пример:

```shell
python mix.py --method exp_time_fade --audios_path audios --save_path smoothed_audio --hifivc_model_path model.pt --vosk_model_path vosk_data --hifigan_model_path cp_hifigan/g_02700000 --config configs/config.json
```

Некоторые параметры установлены по умолчанию, ознакомиться можно в самом [скрипте](./mix.py).

### Обучение модели

Для обучения модели на основе технологии [HIFI-GAN](https://github.com/jik876/hifi-gan.git) используется скрипт [train.py](./train.py), для него предусмотрены следующие флаги:

* --input_wavs_dir - директория с оригинальными (исходными) аудиофайлами;
* --input_mels_dir - директория с mel-спектрограммами для обучения с параметром fine-tuning;
* --input_training_file - текстовый файл, где на каждой строке написан файл, который будет использоваться для обучения модели, пишется без расширения;
* --input_validation_file - текстовый файл, где на каждой строке написан файл, который будет использоваться для валидации модели, пишется без расширения;
* --checkpoint_path - директория, куда будут сохраняться логи и чекпоинты при обучени модели;
* --config - путь до конфигурационного файла;
* --training_epochs - количество эпох обучения;
* --stdout_interval - интервал (шагов) вывода небольшого отчёта о процессе обучения модели;
* --checkpoint_interval - интервал создания чекпоинта модели;
* --summary_interval - интервал логирования процесса обучения модели;
* --validation_interval - интервал валидации модели;
* --fine_tuning - настройка, которой устанавливается значение True, когда для обучения используются собственные mel-спектрограммы;
* --training_data_percent - процент, который используется для обучения модели, 100 - training_data_percent процентов датасета будет использоваться для валидации модели.

Пример:

```shell
python train.py --input_wavs_dir data/validation/origs --input_mels_dir mel_files/spliced --input_training_file data/training.txt --input_validation_file data/validation.txt --checkpoint_path cp_hifigan --config configs/config.json --training_epochs 3100 --stdout_interval 5 --checkpoint_interval 5000 --summary_interval 100 --validation_interval 1000 --fine_tuning True --training_data_percent 80
```

Некоторые параметры установлены по умолчанию, ознакомиться можно в самом [скрипте](./train.py).

Обязательно, чтобы файлы mel-спектрограммы назывались так же, как и оригинальные аудиофайлы, то есть соответственно: orig0.npy -> orig0.wav. В приложении при создании mel-спектрограмм они назвываются в формате orig{number}.npy. Также обязательно, чтобы mel-спектрограммы имели расширение .npy.

### Преобразование аудиофайлов в mel-спектрограммы

Для преобразование аудиофайлов в mel-спектрограммы используется скрипт [mel_spec.py](./mel_spec.py), для него предусмотрены следующие флаги:

* --wavs_dir - директория с аудиофайлами.
* --save_dir - директория, куда требуется сохранить полученные mel-спектрограммы.

Пример:

```shell
python mel_spec.py --wavs_dir data/training/origs --save_dir mel_files/spliced
```
