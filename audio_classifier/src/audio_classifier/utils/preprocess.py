from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Union, Optional
from tqdm import tqdm

import os
import sys
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import essentia.standard as estd


class AudioProcessor(ABC):

    @staticmethod
    def load_audio(audio_file: str, label: str, end_time: int = 1, eq_filter: bool = True):
        """Load and decode a audio file, downmix to mono if necessary and optionally apply Equal loudness filter
         and end time"""
        output = dict()
        # essentia AudioLaoder
        audios, in_sample_rate, n_channels, md5, _, _ = estd.AudioLoader(
            filename=audio_file, computeMD5=True
        )()

        output["sr"] = int(in_sample_rate)
        output["label"] = label
        output["md5"] = md5

        # downmix audio to mono if its multi-channel
        mono_audio = estd.MonoMixer()(audios, n_channels)

        output["len"] = len(mono_audio)
        output["signal"] = mono_audio

        if eq_filter:
            # Apply equal loudness filter
            eq_signal = estd.EqualLoudness(sampleRate=in_sample_rate)(mono_audio)
            output["signal"] = eq_signal

        # Only keep certain sec of audio. If less data do zero padding
        output["signal"] = AudioProcessor.fix_audio_length(output["signal"], size=int((in_sample_rate * end_time) / 2))
        output["len"] = len(output["signal"])
        output["filename"] = os.path.basename(audio_file)

        return output

    @staticmethod
    def fix_audio_length(data, size, axis=-1, **kwargs):
        """Fix audio frames to a given size. Zero pad if necessary"""
        n = data.shape[axis]
        if n > size:
            slices = [slice(None)] * data.ndim
            slices[axis] = slice(0, size)
            return data[tuple(slices)]
        elif n < size:
            lengths = [(0, 0)] * data.ndim
            lengths[axis] = (0, size - n)
            return np.pad(data, lengths, **kwargs)
        return data

    @staticmethod
    def downsample_audio(audio_file: str, label: str, to_samplerate=16000, **kwargs):
        """Downsample audio to a target sample rate"""
        output = AudioProcessor.load_audio(audio_file, label, **kwargs)

        downsampled_audio = estd.Resample(
            inputSampleRate=output["sr"],
            outputSampleRate=to_samplerate
        )(output["signal"])

        output["len"] = len(downsampled_audio)
        output["signal"] = downsampled_audio
        output["sr"] = to_samplerate

        return output


class AudioToTFRecordProcessor(AudioProcessor):
    """Class for processing audio files to a TF record dataset"""
    def __init__(self, audio_dir: str, output_dir: str) -> None:
        self.audio_dir = audio_dir
        self.output_dir = output_dir

    @staticmethod
    def _bytes_feature(value) -> tf.train.Feature:
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value) -> tf.train.Feature:
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value) -> tf.train.Feature:
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def serialize_array(array) -> tf.Tensor:
        array = tf.io.serialize_tensor(array)
        return array

    def get_audio_files(self, allowed_formats: List[str] = [".wav", ".mp3", ".flac", ".ogg"]):
        audios = [audio_file for audio_file in os.listdir(self.audio_dir)
                  if os.path.splitext(audio_file)[1] in allowed_formats]
        return audios

    @staticmethod
    def get_train_validation_split(audio_files: List[str], train_ratio: float = 0.8) -> Tuple:
        """Given a list of audio filenames generate train-validation splits based on a ratio"""
        training_length = int(len(audio_files) * train_ratio)
        validation_length = int(len(audio_files) - training_length)
        shuffled_set = random.sample(audio_files, len(audio_files))
        training_set = shuffled_set[0:training_length]
        validation_set = shuffled_set[-validation_length:]
        return training_set, validation_set

    def parse_audio_feature_to_tf(self, audio_data: Dict) -> tf.train.Example:

        data = {
            'sr': self._int64_feature(audio_data["sr"]),
            'len': self._int64_feature(audio_data["len"]),
            'signal': self._bytes_feature(self.serialize_array(audio_data["signal"])),
            'label': self._int64_feature(int(audio_data["label"])),
            'md5': self._bytes_feature(self.serialize_array(audio_data["md5"])),
            'filename': self._bytes_feature(self.serialize_array(audio_data["filename"]))
        }
        out = tf.train.Example(features=tf.train.Features(feature=data))
        return out

    def to_tf_records(self,
                      audio_files: List[str],
                      dataset_name: str,
                      version: str = "1",
                      split: str = "train",
                      processor_callback=None,
                      **kwargs) -> None:
        """Iterate through audio files, decode and write it to TF record"""
        if not processor_callback:
            processor_callback = self.load_audio

        dataset_path = os.path.join(self.output_dir, dataset_name)
        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(os.path.join(dataset_path, version), exist_ok=True)

        # create a writer that'll store our data to disk
        tf_record_file = os.path.join(dataset_path, version, f"{split}.tfrecords")
        writer = tf.io.TFRecordWriter(tf_record_file)

        with tqdm(
                total=len(audio_files),
                file=sys.stdout,
                desc="audio_classifier:Pre-processing audio files and writing to TFRecords..."
        ) as progress_bar:

            filenames = list()
            md5s = list()
            labels = list()

            for audio_file in audio_files:

                base_name = os.path.basename(audio_file)
                label = base_name.split("_")[0]
                filenames.append(audio_file)
                labels.append(label)

                audio_file = os.path.join(self.audio_dir, audio_file)

                processed_data = processor_callback(audio_file, label=label, **kwargs)
                md5s.append(processed_data["md5"])

                records_out = self.parse_audio_feature_to_tf(processed_data)
                writer.write(records_out.SerializeToString())

                progress_bar.update(1)

            index_df = pd.DataFrame.from_dict({"filename": filenames, "md5": md5s, "label": labels})
            index_df.to_csv(os.path.join(dataset_path, version, f"{split}-index.csv"), index=False)

        writer.close()
