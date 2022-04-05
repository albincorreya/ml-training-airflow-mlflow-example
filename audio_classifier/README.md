# Deep Learning Audio Classifiers

This library is intented to showcase as an example of productionising deep learning ML code to a reusable and modular python library. To start with, we adapt the [Classifiy MNIST Audio Keras example](https://www.kaggle.com/code/christianlillelund/classify-mnist-audio-using-spectrograms-keras-cnn/notebook) from Kaggle.


> Note: By no means this models are closer to state-of-the-art deep learning models for audio classification in literature.


- Modular and configurable design allows to derive new model architecture easily without refactoring 
  code to a certain extent.
- On-the-fly log-scaled mel spectrogram feature extraction layer for ML training (GPU/CPU)
- Ready-to-use audio preprocessing utility functions and TF record dataset loaders
- Easily extendable to different datasets, model architectures and configurations.

## Installation

We use [Poetry](https://python-poetry.org/) for better python dependency management and packaging capabilities.

```bash
# install all dependencies
poetry install
# if you want to avoid development dependencies
poetry install --no-dev
```

## API

- `audio_classifier.models` : custom keras classifier model
- `audio_classifier.layers` : reusable custom keras layers
- `audio_classifier.dataset`: dataset loaders
- `audio_classifier.config`: custom heirarchichal configs
- `audio_classifier.utils.preprocess`: audio raw data preprocessors and TF record writers.


### Scripts

The [scripts](../scripts/ml) directory outlines some example scripts using this library.


- [preprocess_audio.py](../scripts/ml/preprocess_audio.py): an example script for preprocessing raw audio dataset into 
  a structured TF Record audio dataset.

- [train.py](../scripts/ml/train.py): An example model training script

## Tests

The easiest option is to build the playground docker image using docker-compose. 
ie, The Dockerfile takes care of installing dependencies, linting, unit tests and finally 
installing our python package globally in the container.

```
docker-compose build playground
```

> Tip: To use local development process, you might want to setup docker-compose as your remote interpretor of your IDE. 
Check [this](https://www.jetbrains.com/help/pycharm/using-docker-compose-as-a-remote-interpreter.html) 
example of doing it on PyCharm IDE.
> 

Otherwise, cd to this directory and run the following scripts using tox script runner.


- Run flake8 linting

```
tox -e flake8-linting
```

- Run unit tests

```
tox 
```
