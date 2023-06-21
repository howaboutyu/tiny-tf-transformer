import context

import tensorflow as tf
import pytest

from tiny_tf_transformer.transformer import Encoder, Decoder, Transformer
from tiny_tf_transformer.losses_and_metrics import (
    masked_sparse_categorical_accuracy,
    masked_sparse_categorical_cross_entropy,
)


def test_encoder():
    vocab_size = 100
    batch_size = 5
    num_layers = 2
    d_model = 32
    d_ff = 8
    num_heads = 8

    encoder = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        vocab_size=vocab_size,
        num_heads=num_heads,
        d_ff=d_ff,
    )

    seq_length = 123
    x = tf.random.uniform(
        (batch_size, seq_length), minval=0, maxval=vocab_size, dtype=tf.int32
    )

    y = encoder(x)

    assert y.shape == (batch_size, seq_length, d_model)


def test_decoder():
    vocab_size = 100
    batch_size = 5
    num_layers = 2
    d_model = 32
    d_ff = 8
    num_heads = 8

    decoder = Decoder(
        num_layers=num_layers,
        d_model=d_model,
        vocab_size=vocab_size,
        num_heads=num_heads,
        d_ff=d_ff,
    )

    seq_length = 123
    x = tf.random.uniform(
        (batch_size, seq_length), minval=0, maxval=vocab_size, dtype=tf.int32
    )
    encoder_output = tf.random.uniform(
        (batch_size, seq_length, d_model), minval=0, maxval=vocab_size, dtype=tf.float32
    )

    y = decoder(x, encoder_output)

    assert y.shape == (batch_size, seq_length, d_model)
    assert decoder.decoder_layers[-1].cross_attention.attention_scores is not None


def test_transformer():
    input_vocab_size = 100
    target_vocab_size = 200

    batch_size = 5
    num_layers = 2
    d_model = 32
    d_ff = 8
    num_heads = 8

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
    )

    input_seq_length = 123
    target_seq_length = 321

    x = tf.random.uniform(
        (batch_size, input_seq_length),
        minval=0,
        maxval=input_vocab_size,
        dtype=tf.int32,
    )
    y = tf.random.uniform(
        (batch_size, target_seq_length),
        minval=0,
        maxval=target_vocab_size,
        dtype=tf.int32,
    )

    # Test training
    optimizer = tf.keras.optimizers.Adam()
    transformer.compile(
        loss=masked_sparse_categorical_cross_entropy,
        optimizer=optimizer,
        metrics=[masked_sparse_categorical_accuracy],
    )

    transformer.fit(x=[x, y], y=y, epochs=2, batch_size=batch_size)

    y = transformer([x, y])

    assert y.shape == (batch_size, target_seq_length, target_vocab_size)
    assert (
        transformer.decoder.decoder_layers[-1].cross_attention.attention_scores
        is not None
    )
