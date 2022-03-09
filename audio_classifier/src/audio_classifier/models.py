
import tensorflow as tf

from .config import AudioCNNModelConfig
from .layers import MelSpecLayer, AudioCNNPreprocessingLayer
from .layers import AudioCNNFrontEndLayer, AudioCNNEmbeddingLayer, AudioCNNActivationLayer


class AudioCNNModel(tf.keras.Model):
    """Our custom AudioCNN keras model"""
    def __init__(self, config: AudioCNNModelConfig) -> None:
        super(AudioCNNModel, self).__init__()
        # feature extractor and preprocessors (only used for training and validation)
        self.mel_specs = MelSpecLayer(**config.layer_0.dict())
        self.preprocessing = AudioCNNPreprocessingLayer(**config.layer_1.dict())
        # trainable layers
        self.cnn1 = AudioCNNFrontEndLayer(**config.layer_2.dict())
        self.cnn2 = AudioCNNFrontEndLayer(**config.layer_3.dict())
        self.cnn3 = AudioCNNFrontEndLayer(**config.layer_4.dict())
        self.embedding = AudioCNNEmbeddingLayer(**config.layer_5.dict())
        self.activations = AudioCNNActivationLayer(**config.layer_6.dict())

    def call(self, inputs, inference_mode=False, **kwargs) -> tf.Tensor:
        # do feature extraction and preprocessing only during training
        if not inference_mode:
            x = self.mel_specs(inputs)
            x = self.preprocessing(x)
            x = self.cnn1(x)
        else:
            x = self.cnn1(inputs)

        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.embedding(x)
        x = self.activations(x)
        return x

    def build_graph(self):
        x = tf.keras.layers.Input(shape=(None,))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
