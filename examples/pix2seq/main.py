import os
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from omegaconf import OmegaConf
import logging

logging.basicConfig(level=logging.INFO)

from tiny_tf_transformer.transformer import Decoder
from tiny_tf_transformer.losses_and_metrics import (
    masked_sparse_categorical_cross_entropy,
    masked_sparse_categorical_accuracy,
)

from model import get_pix2seq_model, WarmupThenDecaySchedule, get_target_vocab_size
from dataset import preprocess_fn, format_fn, sequence_decoder
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

    def preprocess_and_format(x):
        # Preprocess the data
        preprocessed = preprocess_fn(x, max_side=max_side, num_bins=num_bins)
        # Format the data
        return format_fn(
            *preprocessed,
            sos_token=sos_token,
            eos_token=eos_token,
            max_objects=max_objects,
        )

    # Apply the preprocessing and formatting functions to the datasets
    train_ds = train_ds.map(
        preprocess_and_format, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    val_ds = val_ds.map(
        preprocess_and_format, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # Batch the datasets
    train_ds = train_ds.batch(data_config.batch_size).prefetch(
        tf.data.experimental.AUTOTUNE
    )
    val_ds = val_ds.batch(data_config.batch_size).prefetch(
        tf.data.experimental.AUTOTUNE
    )

    return train_ds, val_ds, sos_token, eos_token


def train(model, train_ds, val_ds, train_config):
    model.fit(
        train_ds,  # .take(train_config.steps_per_epoch),
        epochs=train_config.epochs,
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
        target_vocab_size=get_target_vocab_size(num_bins),
        attention_dropout_rate=model_config.attention_dropout_rate,
        ff_dropout_rate=model_config.ff_dropout_rate,
        max_length=max_objects * 5,  # num_objects * len([x, y, w, h, label])
    )

    lr_schedule = WarmupThenDecaySchedule(
        config.training.initial_learning_rate,
        config.training.epochs,
        config.training.warmup_epochs,
        config.training.steps_per_epoch,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        loss=masked_sparse_categorical_cross_entropy,
        optimizer=optimizer,
        metrics=[masked_sparse_categorical_accuracy],
    )

    return model


def eval(model, val_ds, config):
    batch_id = 0
    for (image_batch, y_tok_in), y_tok_out in val_ds.take(1):
        y_tok_pred_probs = model([image_batch, y_tok_in])
        y_tok_pred = np.argmax(y_tok_pred_probs, -1)
        bboxes, labels = sequence_decoder(
            y_tok_pred, eos_token=config.eos_token, num_bins=config.data_config.num_bins
        )

        for i in range(config.data_config.batch_size):
            bboxes_np = np.asarray(bboxes[i]) * config.data_config.max_side
            labels_np = np.asarray(labels[i])

            scores = np.ones_like(labels_np).astype(np.float32)
            os.makedirs("eval_images", exist_ok=True)
            image_path = f"eval_images/batch_{batch_id}_{i}.png"
            visualize_detections(
                image_batch[i], bboxes_np, labels_np, scores, save_path=image_path
            )


def main():
    config = OmegaConf.load("config.yaml")

    train_ds, val_ds, sos_token, eos_token = get_dataset(config.data_config)
    config.sos_token = sos_token
    config.eos_token = eos_token

    logging.info(f"Start token: {sos_token}")
    logging.info(f"End token: {eos_token} ")

    model = get_model(config)

    # load weights
    if os.path.exists(config.checkpoint + ".index"):
        model.load_weights(config.checkpoint)
        logging.info(f"Loaded weights from {config.checkpoint}")
    else:
        logging.warning(f"No checkpoints found at {config.checkpoint}")

    train(model, train_ds, val_ds, config.training)
    eval(model, train_ds, config)

    assert (
        config.checkpoint_output_dir
    ), "Need to specify newly trained output model path."

    model.save_weights(config.checkpoint_output_dir)
    logging.info(f"Saved model at :{config.checkpoint_output_dir}")


if __name__ == "__main__":
    main()
