
## Data Source

We use the [free-spoken-digit dataset](https://zenodo.org/record/1342401) for our dummy audio ML training job.


Eg: you use the following bash script at this directory level to download the raw audio files of dataset from Zenodo.

> make sure you're on ./data path while running the script
```bash
#!/bin/bash
curl https://zenodo.org/record/1342401/files/Jakobovski/free-spoken-digit-dataset-v1.0.8.zip -OJL

dataSource=$(find . -name 'Jakobovski-free-spoken-digit-dataset*')
cp -r $dataSource/recordings ./
rm -r $dataSource
```

## Data preprocessing

Data preprocessing is done by the [preprocess_audio.py](../scripts/ml/preprocess_audio.py) script.