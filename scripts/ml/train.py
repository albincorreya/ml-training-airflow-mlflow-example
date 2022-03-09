import os
import uuid
import click
import mlflow
import tensorflow as tf

from audio_classifier.dataset import RecordingsDataset
from audio_classifier.config import AudioCNNYAMLConfig
from audio_classifier.models import AudioCNNModel


@click.command()
@click.option('--dataset-path', default="/workspace/data/datasets/MNIST-audio/1",
              help='Root path to the TF record dataset')
@click.option('--n-epochs', default=2, help='No. of epochs for model training')
@click.option('--data-batch-size', default=32, help='Batch size for model training')
@click.option('--n-classes', default=10, help='Number of classes in the dataset to be trained')
@click.option('--correlation-id', default=uuid.uuid4(), help='Correlation ID for the training job')
@click.option('--model-yaml-config', default="/workspace/scripts/ml/configs/config-1.yaml",
              help='Path to custom yaml model config file')
def train(
        dataset_path: str,
        n_epochs: int,
        data_batch_size: int,
        n_classes: int,
        correlation_id: str,
        model_yaml_config: str):

    # Enable tracking to our remote MLFlow server
    mlflow.set_tracking_uri(os.getenv("MLFLOW_REMOTE_TRACKING_URI"))
    mlflow.tensorflow.autolog()
    # log function args
    mlflow.log_params(locals())
    # log model config artifact
    mlflow.log_artifact(model_yaml_config)
    # default tag
    mlflow.set_tags({
        'environment': 'testing'
    })

    # Load dataset
    train_set = RecordingsDataset(os.path.join(dataset_path, "train.tfrecords"))
    validation_set = RecordingsDataset(os.path.join(dataset_path, "validation.tfrecords"))

    # shuffle and get dataset as batches
    train_set = train_set.get_dataset_as_batches(batch_size=data_batch_size)
    validation_set = validation_set.get_dataset_as_batches(batch_size=data_batch_size)

    # Read the custom YAML model config from file
    args = AudioCNNYAMLConfig(model_yaml_config).to_args()
    # custom audio cnn model
    model = AudioCNNModel(config=args)
    # do a forward pass to build the graph first
    model = model.build_graph()
    # model architecture summary
    print(model.summary())

    # get mlflow artifact path
    artifact_path = mlflow.get_artifact_uri()
    os.system(f"cp {model_yaml_config} {artifact_path}/")

    # callback for storing the best weights
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(artifact_path, "checkpoint.h5"),
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'
    )
    # compile model with opts
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.RMSprop(),
        metrics=["accuracy"]
    )
    # Run training
    history = model.fit(
        train_set,
        epochs=n_epochs,
        validation_data=validation_set,
        batch_size=data_batch_size,
        callbacks=[checkpoint]
    )
    # Run evaluation
    metrics = model.evaluate(validation_set, verbose=0)

    # Save model
    model.save(
        os.path.join(artifact_path, "model.h5"),
        include_optimizer=True,
        save_format="h5"
    )


if __name__ == "__main__":
    train()
