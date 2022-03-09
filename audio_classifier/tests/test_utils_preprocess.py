import pytest


@pytest.mark.data
def test_get_files_pass(audio_to_tfrecord_processor):
    assert audio_to_tfrecord_processor.get_audio_files()


@pytest.mark.data
def test_get_dataset_split_not_empty(audio_to_tfrecord_processor):
    files = audio_to_tfrecord_processor.get_audio_files()
    train, validation = audio_to_tfrecord_processor.get_train_validation_split(files, train_ratio=0.5)
    assert train and validation


@pytest.mark.data
def test_process_tf_record_not_fails(audio_to_tfrecord_processor, test_dataset_dir):
    files = audio_to_tfrecord_processor.get_audio_files()
    audio_to_tfrecord_processor.to_tf_records(files, dataset_name="test")
