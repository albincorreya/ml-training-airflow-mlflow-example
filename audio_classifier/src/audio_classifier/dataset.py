
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Union

import tensorflow as tf


class BaseAudioDataset(ABC):

    # schema for audio TF records
    tfr_schema: Dict = {
        'sr': tf.io.FixedLenFeature([], tf.int64),
        'len': tf.io.FixedLenFeature([], tf.int64),
        'signal': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'md5': tf.io.FixedLenFeature([], tf.string),
        'filename': tf.io.FixedLenFeature([], tf.string),
    }

    @abstractmethod
    def load(self, **kwargs) -> tf.data.Dataset:
        pass

    @staticmethod
    def _parse_tfr_all_audio_element(element) -> Tuple:
        """Parse all the columns inside a single TFRecord element of BaseAudioDataset"""
        sample = tf.io.parse_single_example(element, BaseAudioDataset.tfr_schema)
        # unpack sample tuple
        sample_rate = sample['sr']
        audio_length = sample['len']
        audio_signal = sample['signal']
        label = sample['label']
        md5 = sample['md5']
        filename = sample["filename"]
        # parse audio signal data
        audio_signal = tf.io.parse_tensor(audio_signal, out_type=tf.float32)
        audio_signal = tf.reshape(audio_signal, shape=[audio_length])

        return audio_signal, sample_rate, label, md5, filename

    @staticmethod
    def _parse_tfr_audio_label_element(element) -> Tuple:
        """Parse only the audio signal and label data from a single TFRecord element of BaseAudioDataset"""
        audio_signal, _, label, _, _ = BaseAudioDataset._parse_tfr_all_audio_element(element)
        return audio_signal, label

    @staticmethod
    def _load_dataset(filename: str, parser_func) -> tf.data.Dataset:
        """Load the TFRecord dataset given a predefined parser_fun"""
        # create the dataset
        dataset = tf.data.TFRecordDataset(filename)
        # pass every single feature through our mapping function
        dataset = dataset.map(
            parser_func
        )
        return dataset


class RecordingsDataset(BaseAudioDataset):
    def __init__(self, filename: str, full_schema: bool = False, **kwargs):
        self.filename = filename
        self.dataset = self.load(full_schema=full_schema)

    def load(self, full_schema: bool = False) -> tf.data.Dataset:
        """Load the dataset"""
        if full_schema:
            return self._load_dataset(self.filename, self._parse_tfr_all_audio_element)
        else:
            return self._load_dataset(self.filename, self._parse_tfr_audio_label_element)

    def get_dataset_as_batches(self, batch_size: int = 32):
        """Get dataset as batches for training jobs with given parameters"""
        dataset = self.dataset.shuffle(2048)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset.batch(batch_size)
