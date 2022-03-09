

def test_audiocnn_config_name_valid(audio_cnn_yaml_config):
    assert audio_cnn_yaml_config.config["model_name"] == "AudioCNNModel"


def test_audiocnn_config_schema_valid(audio_cnn_yaml_config):
    print(audio_cnn_yaml_config.to_args().dict())
    assert len(audio_cnn_yaml_config.to_args().dict().keys()) == 7
