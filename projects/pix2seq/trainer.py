from dataset import preprocess_fn, format_fn
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

from tiny_tf_transformer.transformer import Decoder
from tiny_tf_transformer.losses_and_metrics import (
    masked_sparse_categorical_cross_entropy,
    masked_sparse_categorical_accuracy,
)

from model import get_pix2seq_model

(train_ds, val_ds), dataset_info = tfds.load(
    "coco/2017", split=["train", "validation"], with_info=True, data_dir="~/data"
)

max_side = 128
num_bins = 128
batch_size = 2
max_objects = 10

EOS_TOKEN = num_bins + 1

train_ds = train_ds.map(
    lambda x: preprocess_fn(x, max_side=max_side, num_bins=num_bins),
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
)

train_ds = train_ds.map(
    lambda image, bboxes, label, image_shape: format_fn(
        image, bboxes, label, image_shape, EOS_TOKEN, max_objects=max_objects
    ),
)

train_ds = train_ds.batch(batch_size)

for t in train_ds.take(1):
    (image, decoder_input), decoder_output = t
    import pdb; pdb.set_trace()
    print(image.shape)
    print(decoder_input.shape)
    print(decoder_output.shape)




input_shape = (max_side, max_side, 3)

model = get_pix2seq_model(
    input_shape,
    model_name="resnet50v2",
    num_layers=4,
    d_model=128,
    num_heads=8,
    d_ff=512,
    target_vocab_size=num_bins + 2,
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
    train_ds.take(10),
    epochs=100,
    validation_data=train_ds.take(1),
)


