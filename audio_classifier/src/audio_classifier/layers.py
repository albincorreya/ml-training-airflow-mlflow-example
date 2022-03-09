import tensorflow as tf
import tensorflow_io as tfio


class AudioCNNFrontEndLayer(tf.keras.layers.Layer):
    """Custom front-end CNN layer with pooling and batch norm for AudioCNn model"""
    def __init__(self,
                 n_filters=32,
                 kernal_size=3,
                 strides=(1, 1),
                 activation="relu",
                 **kwargs) -> None:
        super(AudioCNNFrontEndLayer, self).__init__()
        # params
        self.n_filters = n_filters
        self.kernal_size = kernal_size
        self.strides = strides
        self.activation = activation

        # define layers
        self.conv2d = tf.keras.layers.Conv2D(
            self.n_filters,
            self.kernal_size,
            strides=self.strides,
            padding="same",
            activation=self.activation
        )
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.norm2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs) -> tf.Tensor:
        x = self.conv2d(inputs)
        x = self.norm1(x)
        x = self.pool(x)
        x = self.norm2(x)
        return x

    def get_config(self):
        config = super(AudioCNNFrontEndLayer, self).get_config()
        config.update(
            {
                "n_filters": self.n_filters,
                "kernal_size": self.kernal_size,
                "strides": self.strides,
                "activation": self.activation
            }
        )
        return config


class AudioCNNEmbeddingLayer(tf.keras.layers.Layer):
    """Custom dense layer for trained embeddings of AudioCNN model"""
    def __init__(self, num_dense_layer=10, activation="relu") -> None:
        super(AudioCNNEmbeddingLayer, self).__init__()
        # params
        self.num_dense_layer = num_dense_layer
        self.activation = activation

        # define layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(256, activation="relu")

    def call(self, inputs) -> tf.Tensor:
        x = self.flatten(inputs)
        x = self.dense(x)
        return x

    def get_config(self):
        config = super(AudioCNNEmbeddingLayer, self).get_config()
        config.update(
            {
                "num_dense_layer": self.num_dense_layer,
                "activation": self.activation
            }
        )
        return config


class AudioCNNActivationLayer(tf.keras.layers.Layer):
    """Custom activation layer for AudioCNN model"""
    def __init__(self, dropout_rate=0.5, num_dense_layer=10, activation="softmax", **kwargs) -> None:
        super(AudioCNNActivationLayer, self).__init__()
        # params
        self.dropout_rate = dropout_rate
        self.num_dense_layer = num_dense_layer
        self.activation = activation

        # define layers
        self.norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense = tf.keras.layers.Dense(self.num_dense_layer, activation=self.activation)

    def call(self, inputs) -> tf.Tensor:
        x = self.norm(inputs)
        x = self.dropout(x)
        x = self.dense(x)
        return x

    def get_config(self):
        config = super(AudioCNNActivationLayer, self).get_config()
        config.update(
            {
                "dropout_rate": self.dropout_rate,
                "num_dense_layer": self.num_dense_layer,
                "activation": self.activation
            }
        )
        return config


class AudioCNNPreprocessingLayer(tf.keras.layers.Layer):
    """Layer to preprocess and augment input log mel spectrogram into image"""
    def __init__(self, image_height: int = 256, image_width: int = 256):
        super(AudioCNNPreprocessingLayer, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        # keras preprocessing layers
        self.rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)
        self.flip = tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")
        self.rotate = tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)

    def to_image(self, log_mel_specs):
        """Convert log mel spectogram into image"""
        # expand one dimension axis to be treated as image
        image = tf.expand_dims(log_mel_specs, axis=-1)
        image = tf.image.resize(image, (self.image_height, self.image_width))
        image = tf.image.per_image_standardization(image)
        # no augmentation at this stage
        image = (image - tf.reduce_mean(image)) / (tf.reduce_max(image) * tf.reduce_min(image)) * 255.0
        image = tf.image.grayscale_to_rgb(image)
        return image

    def call(self, inputs, training=True):
        x = self.to_image(inputs)
        x = self.rescale(x)
        # only do augmentation during training
        if training:
            x = self.flip(x)
            x = self.rotate(x)
        return x

    def get_config(self):
        config = super(AudioCNNPreprocessingLayer, self).get_config()
        config.update(
            {
                "image_height": self.image_height,
                "image_width": self.image_width,
            }
        )
        return config


class MelSpecLayer(tf.keras.layers.Layer):
    """Layer to compute Log mel spectrogram from an audio input"""
    def __init__(
            self,
            frame_length=512,
            frame_step=256,
            fft_length=None,
            sampling_rate=8000,
            num_mel_bands=256,
            freq_min=40,
            freq_max=4000,
            **kwargs,
    ):
        super(MelSpecLayer, self).__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.sampling_rate = sampling_rate
        self.num_mel_bands = num_mel_bands
        self.freq_min = freq_min
        self.freq_max = freq_max
        # Defining mel filter. This filter will be multiplied with the STFT output
        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mel_bands,
            num_spectrogram_bins=self.frame_length // 2 + 1,
            sample_rate=self.sampling_rate,
            lower_edge_hertz=self.freq_min,
            upper_edge_hertz=self.freq_max,
        )

    def call(self, audio):
        # We will only perform the transformation during training.
        stft = tf.signal.stft(
            audio,
            self.frame_length,
            self.frame_step,
            self.fft_length,
            pad_end=True,
        )
        # Taking the magnitude of the STFT output
        magnitude = tf.abs(stft)
        # Multiplying the Mel-filterbank with the magnitude and scaling it using the db scale
        mel = tf.matmul(tf.square(magnitude), self.mel_filterbank)
        log_mel_spec = tfio.audio.dbscale(mel, top_db=80)

        return log_mel_spec

    def get_config(self):
        config = super(MelSpecLayer, self).get_config()
        config.update(
            {
                "frame_length": self.frame_length,
                "frame_step": self.frame_step,
                "fft_length": self.fft_length,
                "sampling_rate": self.sampling_rate,
                "num_mel_bands": self.num_mel_bands,
                "freq_min": self.freq_min,
                "freq_max": self.freq_max,
            }
        )
        return config
