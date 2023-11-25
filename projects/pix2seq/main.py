from dataset import preprocess_fn, format_fn, PAD_VALUE
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

from tiny_tf_transformer.transformer import Decoder
from tiny_tf_transformer.losses_and_metrics import (
    masked_sparse_categorical_cross_entropy,
    masked_sparse_categorical_accuracy,
)

from model import get_pix2seq_model





def get_dataset(batch_size=32, max_side=512, num_bins=256, max_objects=20):
    (train_ds, val_ds), dataset_info = tfds.load(
        "coco/2017", split=["train", "validation"], with_info=True, data_dir="~/data"
    )

    EOS_TOKEN = num_bins + 1

    train_ds = train_ds.map(
        lambda x: preprocess_fn(x, max_side=max_side, num_bins=num_bins),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    val_ds = val_ds.map(
        lambda x: preprocess_fn(x, max_side=max_side, num_bins=num_bins),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    train_ds =train_ds.map(
        lambda image, bboxes, label, image_shape: format_fn(
            image, bboxes, label, image_shape, EOS_TOKEN, max_objects=max_objects
        ),
    )

    val_ds =val_ds.map(
        lambda image, bboxes, label, image_shape: format_fn(
            image, bboxes, label, image_shape, EOS_TOKEN, max_objects=max_objects
        ),
    )

    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)



    return train_ds, val_ds

def train(train_ds, val_ds, max_side=512, num_bins=256, max_objects=20):
    
    input_shape = (max_side, max_side, 3)

    model = get_pix2seq_model(
        input_shape,
        model_name="resnet50v2",
        num_layers=4,
        d_model=128,
        num_heads=8,
        d_ff=512,
        target_vocab_size=num_bins + 4,
        attention_dropout_rate=0.1,
        ff_dropout_rate=0.1,
        max_length=max_objects * 5,
    )

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



if __name__ == '__main__':
    train_ds, val_ds = get_dataset()

    for (x, y_in), y_out in train_ds.take(1):
        print(y_out)
        print(y_out)
        import pdb; pdb.set_trace()

    train(train_ds, val_ds)