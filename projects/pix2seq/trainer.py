from dataset import preprocess_fn, format_fn 
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

from tiny_tf_transformer.transformer import Decoder
from tiny_tf_transformer.losses_and_metrics import (
    masked_sparse_categorical_cross_entropy,
    masked_sparse_categorical_accuracy,
    )

(train_ds, val_ds), dataset_info = tfds.load( "coco/2017", split=["train", "validation"], with_info=True, data_dir="~/data" )

max_side = 128 
num_bins = 128 
batch_size = 2 
max_objects =10 

EOS_TOKEN = num_bins + 1

train_ds = train_ds.map( 
    lambda x: preprocess_fn(x, max_side=max_side, num_bins=num_bins), 
    num_parallel_calls=tf.data.experimental.AUTOTUNE )

train_ds = train_ds.map(
    lambda image, bboxes, label, image_shape: format_fn(image, bboxes, label, image_shape, EOS_TOKEN, max_objects=max_objects),
    )

train_ds = train_ds.batch( batch_size )




def get_model(
        input_shape, 
        model_name="resnet50v2",
        num_layers: int = 4,
        d_model: int = 128,
        num_heads: int = 8,
        d_ff: int = 512,
        target_vocab_size: int = 256,
        attention_dropout_rate: float = 0.1,
        ff_dropout_rate: float = 0.1,
        max_length: int = 2048,

        ):

    if model_name == "resnet50v2":
        feature_extractor = tf.keras.applications.ResNet50V2( include_top=False, weights="imagenet" )   
        preprocess_fn = tf.keras.applications.resnet_v2.preprocess_input

    input = tf.keras.layers.Input( shape=input_shape )
    decoder_input = tf.keras.layers.Input( shape=(max_length,) )

    x = preprocess_fn( input )

    x = feature_extractor(x)

    decoder = Decoder(
        num_layers=num_layers,
        d_model=d_model,
        vocab_size=target_vocab_size,
        num_heads=num_heads,
        d_ff=d_ff,
        attention_dropout_rate=attention_dropout_rate,
        ff_dropout_rate=ff_dropout_rate,
        max_length=max_length,
        )

    final_layer = tf.keras.layers.Dense(target_vocab_size, name="final_layer")

    # reshape context from [n, n, d] to [n * n, d]
    x = tf.keras.layers.Reshape( (-1, x.shape[-1]) )(x)

    # reduce dimensionality
    x = tf.keras.layers.Dense( d_model, activation="relu" )(x) 

    x = decoder( decoder_input, x ) 

    x = final_layer(x)

    model = tf.keras.Model( inputs=[input, decoder_input], outputs=x )

    print(model.summary())
    return model



input_shape = (max_side, max_side, 3)

model = get_model(
    input_shape,
    model_name="resnet50v2",
    num_layers=4,
    d_model=128,
    num_heads=8,
    d_ff=512,
    target_vocab_size=num_bins+2,
    attention_dropout_rate=0.1,
    ff_dropout_rate=0.1,
    max_length=max_objects*5,
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
    train_ds.take(1000),
    epochs=10,
    validation_data=train_ds.take(1),
    )


for t in train_ds.take(10):
    (image, decoder_input), decoder_output = t
    print(np.min(decoder_output))
    #print(image.shape)
    #print(decoder_output.shape)
    #print(decoder_input.shape)



