import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from omegaconf import OmegaConf

from tiny_tf_transformer.transformer import Decoder
from tiny_tf_transformer.losses_and_metrics import (
    masked_sparse_categorical_cross_entropy,
    masked_sparse_categorical_accuracy,
)

from model import get_pix2seq_model
from dataset import preprocess_fn, format_fn, PAD_VALUE, sequence_decoder
from utils import visualize_detections


def get_dataset(data_config):
    # Load the dataset using TensorFlow Datasets
    (train_ds, val_ds), dataset_info = tfds.load(
        "coco/2017",
        split=["train", "validation"],
        with_info=True,
        data_dir=data_config.data_dir,  # Use data directory from the config
    )

    # Define special tokens based on num_bins from the config
    sos_token = data_config.num_bins + 1
    eos_token = data_config.num_bins + 2

    max_side = data_config.max_side
    num_bins = data_config.num_bins
    max_objects = data_config.max_objects

    # Function to apply preprocessing and formatting
    def preprocess_and_format(x):
        # Preprocess the data
        preprocessed = preprocess_fn(x, max_side=max_side, num_bins=num_bins)
        # Format the data
        return format_fn(
            *preprocessed,
            sos_token=sos_token,
            eos_token=eos_token,
            max_objects=max_objects
        )

    # Apply the preprocessing and formatting functions to the datasets
    train_ds = train_ds.map(
        preprocess_and_format, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    val_ds = val_ds.map(
        preprocess_and_format, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # Batch the datasets
    train_ds = train_ds.batch(data_config.batch_size)
    val_ds = val_ds.batch(data_config.batch_size)

    return train_ds, val_ds, sos_token, eos_token


def train(train_ds, val_ds, max_side=512, num_bins=256, max_objects=20):
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4,
    )

    model.compile(
        loss=masked_sparse_categorical_cross_entropy,
        optimizer=optimizer,
        metrics=[masked_sparse_categorical_accuracy],
    )

    model.fit(
        train_ds.take(1),
        epochs=100,
        validation_data=val_ds.take(1),
    )


def get_model(config):
    # Retrieve model and data parameters from the configuration
    max_side = config.data_config.max_side
    num_bins = config.data_config.num_bins
    max_objects = config.data_config.max_objects

    # Model configuration
    model_config = config.model

    # Define the input shape
    input_shape = (max_side, max_side, 3)

    # Create the model using parameters from the configuration
    model = get_pix2seq_model(
        input_shape,
        model_name=model_config.name,
        num_layers=model_config.num_layers,
        d_model=model_config.d_model,
        num_heads=model_config.num_heads,
        d_ff=model_config.d_ff,
        target_vocab_size=num_bins + 4,  # Adjusted for special tokens
        attention_dropout_rate=model_config.attention_dropout_rate,
        ff_dropout_rate=model_config.ff_dropout_rate,
        max_length=max_objects * 5,  # Assuming 5 tokens per object
    )

    return model


if __name__ == "__main__":
    config = OmegaConf.load("config.yaml")

    train_ds, val_ds, sos_token, eos_token = get_dataset(config.data_config)

    model = get_model(config)

    for (x, y_in), y_out in train_ds.take(1):
        bboxes, labels = sequence_decoder(
            y_out, eos_token=eos_token, num_bins=config.data_config.num_bins
        )
        images = x

        for i in range(config.data_config.batch_size):
            bboxes_np = np.asarray(bboxes[i]) * config.data_config.max_side
            labels_np = np.asarray(labels[i])

            scores = np.ones_like(labels_np).astype(np.float32)

            visualize_detections(images[i], bboxes_np, labels_np, scores)

    train(train_ds, val_ds)
