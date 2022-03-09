import pytest
from audio_classifier.models import AudioCNNModel
from audio_classifier.config import AudioCNNYAMLConfig
from audio_classifier.utils.preprocess import AudioToTFRecordProcessor


def pytest_configure(config):
    """
    Performs initial configuration of the session. Official docs:
    https://docs.pytest.org/en/stable/reference.html#pytest.hookspec.pytest_configure

    Example usage:

    - @pytest.mark.ml
    - @pytest.mark.data
    """
    # we might want to avoid running tests related to triton client in the BB pipeline and test it locally instead.
    config.addinivalue_line(
        "markers", "ml: mark tests related to ml part of library"
    )
    config.addinivalue_line(
        "markers", "data: mark tests related to data processing"
    )


@pytest.fixture(scope="session")
def audio_dir():
    return "/workspace/data/recordings"


@pytest.fixture(scope="session")
def test_dataset_dir():
    return "tests/data/datasets"


@pytest.fixture(scope="session")
def test_model_config_file():
    return "tests/data/audiocnn_config.yaml"


@pytest.fixture(scope="session")
def audio_cnn_yaml_config(test_model_config_file):
    return AudioCNNYAMLConfig(test_model_config_file)


@pytest.fixture(scope="session")
def audio_to_tfrecord_processor(audio_dir, test_dataset_dir):
    return AudioToTFRecordProcessor(
        audio_dir=audio_dir,
        output_dir=test_dataset_dir
    )


@pytest.fixture(scope="session")
def audio_cnn_model(audio_cnn_yaml_config):
    return AudioCNNModel(config=audio_cnn_yaml_config.to_args())
