import pytest
import tensorflow as tf
from audio_classifier.layers import MelSpecLayer, AudioCNNPreprocessingLayer
from audio_classifier.layers import AudioCNNFrontEndLayer, AudioCNNEmbeddingLayer, AudioCNNActivationLayer


@pytest.mark.ml
def test_mel_spec_layer_call():
    x = tf.ones(shape=(500,), dtype=tf.float32)
    layer = MelSpecLayer()
    x = layer(x)


@pytest.mark.ml
def test_audio_cnn_preporcess_layer_pass_call():
    x = tf.ones(shape=(17, 256), dtype=tf.float32)
    layer = AudioCNNPreprocessingLayer()
    x = layer(x)


@pytest.mark.ml
def test_audio_cnn_frontend_layer_pass_compute():
    x = tf.ones(shape=(1, 256, 256, 3), dtype=tf.float32)
    layer = AudioCNNFrontEndLayer(32, 3, strides=2)
    x = layer(x)


def test_audio_cnn_emdedding_layer_pass_compute():
    x = tf.ones(shape=(1, 16, 16, 128), dtype=tf.float32)
    layer = AudioCNNEmbeddingLayer()
    x = layer(x)


def test_audio_cnn_backend_layer_pass_compute():
    x = tf.ones(shape=(1, 10), dtype=tf.float32)
    layer = AudioCNNActivationLayer()
    x = layer(x)
