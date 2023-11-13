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


(train_ds, val_ds), dataset_info = tfds.load(
    "coco/2017", split=["train", "validation"], with_info=True, data_dir="~/data"
)

import pdb; pdb.set_trace()
max_side = 256 
num_bins = 256 
batch_size = 32 
max_objects = 20

EOS_TOKEN = num_bins + 1

train_ds = train_ds.map(
    lambda x: preprocess_fn(x, max_side=max_side, num_bins=num_bins),
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
)


val_ds = val_ds.map(
    lambda x: preprocess_fn(x, max_side=max_side, num_bins=num_bins),
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
)


#train_ds = train_ds.shuffle(1024)

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


#for (a, b), c in train_ds.take(10000):
#    import pdb; pdb.set_trace()
#    print('---')
#    #print(f'max {np.max(b[:, #:4])}')
#    max_ = np.max(b)
#    print(max_)
#    if max_ > 1:
#        import pdb; pdb.set_trace()
#
#
#
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

#loss = lambda y_true, y_pred: masked_sparse_categorical_cross_entropy(y_true, y_pred,  PAD_VALUE)
accuracy = lambda y_true, y_pred: masked_sparse_categorical_accuracy(y_true, y_pred, PAD_VALUE) 


def loss(y_true, y_pred):
    return masked_sparse_categorical_cross_entropy(y_true, y_pred,  PAD_VALUE)



#loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    loss=loss,
    optimizer=optimizer,
    metrics=[accuracy],
)

model.fit(
    train_ds.take(10),
    epochs=100,
    validation_data=val_ds.take(1),
)


