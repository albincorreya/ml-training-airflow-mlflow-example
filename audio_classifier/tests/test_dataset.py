from audio_classifier.dataset import RecordingsDataset


def test_load_recordings_dataset_succeed(test_dataset_dir):
    _ = RecordingsDataset(test_dataset_dir)


def test_load_recordings_dataset_with_full_schema_succeed(test_dataset_dir):
    _ = RecordingsDataset(test_dataset_dir, full_schema=True)


def test_get_dataset_as_batches_succeed(test_dataset_dir):
    dataset = RecordingsDataset(test_dataset_dir)
    _ = dataset.get_dataset_as_batches(batch_size=16)

