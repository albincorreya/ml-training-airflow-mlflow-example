
import click
from audio_classifier.utils.preprocess import AudioToTFRecordProcessor


@click.command()
@click.option('--dataset-name', default="MNIST-audio", help='Name of dataset')
@click.option('--dataset-version', help='Dataset version')
@click.option('--audio-dir', help='Path to directory of audio files')
@click.option('--output-dir', help='Path to directory where you want to output the tf records')
@click.option('--train-split', default=0.8, help='Ratio of files for the train split')
def process(
        dataset_name: str,
        dataset_version: int,
        audio_dir: str,
        output_dir: str,
        train_split: float):

    # Create a TF record process for loading,decoding, preprocessing and storing audio files
    processor = AudioToTFRecordProcessor(
        audio_dir=audio_dir,
        output_dir=output_dir
    )
    # get all valid audio files in the audio directory
    files = processor.get_audio_files()
    # get a train-validation split of raw dataset
    train_split, validation_split = processor.get_train_validation_split(files, train_ratio=train_split)
    splits = [{"name": "train", "files": train_split}, {"name": "validation", "files": validation_split}]
    # Write TF record file for each data split
    for data in splits:
        print(f"Processing {data['name']} split of dataset")
        processor.to_tf_records(
            audio_files=data["files"],
            split=data["name"],
            dataset_name=dataset_name,
            version=str(dataset_version),
            processor_callback=processor.load_audio
        )
    print("Done...")


if __name__ == "__main__":
    process()
