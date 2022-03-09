
import yaml
from pydantic import BaseModel
from typing import Any, Optional, Dict


class MelSpecLayerConfig(BaseModel):
    sampling_rate: int
    num_mel_bands: int


class AudioCNNPreprocessingLayerConfig(BaseModel):
    image_height: int
    image_width: int


class AudioCNNFrontEndLayerConfig(BaseModel):
    n_filters: Any
    kernal_size: int
    strides: Optional[Any] = (1, 1)
    activation: str


class AudioCNNEmbeddingLayerConfig(BaseModel):
    num_dense_layer: int
    activation: str


class AudioCNNBackendLayerConfig(BaseModel):
    dropout_rate: float
    num_dense_layer: int
    activation: str


class AudioCNNModelConfig(BaseModel):
    layer_0: MelSpecLayerConfig
    layer_1: AudioCNNPreprocessingLayerConfig
    layer_2: AudioCNNFrontEndLayerConfig
    layer_3: AudioCNNFrontEndLayerConfig
    layer_4: AudioCNNFrontEndLayerConfig
    layer_5: AudioCNNEmbeddingLayerConfig
    layer_6: AudioCNNBackendLayerConfig


class YAMLConfig:
    """Base class for yaml config"""
    def __init__(self, filename: str):
        self.filename = filename
        self.config = self.load(filename)

    @staticmethod
    def load(filename: str) -> Dict:
        data = None
        with open(filename, "r") as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as err:
                raise IOError(err)
        return data

    def parse_layers_to_model_schema(self) -> Dict:
        config = dict()
        for index, layer in enumerate(self.config["layers"]):
            config[f"layer_{index}"] = layer
        return config


class AudioCNNYAMLConfig(YAMLConfig):
    """Class for loading config for audio cnn model"""
    def __init__(self, filename: str):
        super(AudioCNNYAMLConfig, self).__init__(filename)

    def to_args(self) -> AudioCNNModelConfig:
        config = self.parse_layers_to_model_schema()
        return AudioCNNModelConfig(**config)
