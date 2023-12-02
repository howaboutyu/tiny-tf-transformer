import context

import tensorflow as tf

from tiny_tf_transformer.embedding_layers import (
    PositionalTokenEmbedding,
    PositionalEmbedding,
)

from tiny_tf_transformer.transformer_layers import (
    BaseAttention,
    CausalAttention,
    CrossAttention,
    SelfAttention,
    FeedFoward,
    EncoderLayer,
    DecoderLayer,
)


def test_positional_embedding():
    d_model = 128
    max_length = 100
    vocal_size = 10000

    x = tf.random.uniform(
        (1, max_length - 10, 1), minval=0, maxval=vocal_size, dtype=tf.float32
    )
    x = tf.concat([x, tf.zeros((1, 10, 1), dtype=tf.float32)], axis=1)

    # -- test with float32
    pos_embedding = PositionalEmbedding(d_model, max_length, tf.float32)
    y = pos_embedding(x)

    assert y.shape == (1, max_length, d_model)
    assert y.dtype == tf.float32

    # -- test with float16
    pos_embedding = PositionalEmbedding(d_model, max_length, tf.float16)
    y = pos_embedding(x)

    assert y.shape == (1, max_length, d_model)
    assert y.dtype == tf.float16

    # -- test mask
    mask = pos_embedding.compute_mask(x)

    assert tf.reduce_sum(tf.cast(mask, tf.int32)) == 90


def test_positional_token_embedding():
    d_model = 128
    max_length = 100
    vocal_size = 10000

    x = tf.random.uniform((1, 90), minval=0, maxval=vocal_size, dtype=tf.int32)
    x = tf.concat([x, tf.zeros((1, 10), dtype=tf.int32)], axis=1)

    pos_token_embedding = PositionalTokenEmbedding(
        vocal_size, d_model, max_length, tf.float32
    )
    y = pos_token_embedding(x)

    assert y.shape == (1, 100, 128)
    assert y.dtype == tf.float32

    pos_token_embedding = PositionalTokenEmbedding(
        vocal_size, d_model, max_length, tf.float16
    )
    y = pos_token_embedding(x)

    assert y.shape == (1, 100, 128)
    assert y.dtype == tf.float16

    # -- test mask
    mask = pos_token_embedding.compute_mask(x)

    assert tf.reduce_sum(tf.cast(mask, tf.int32)) == 90


def test_attention_layers():
    num_heads = 2
    key_dim = 64

    base_attention = BaseAttention(num_heads=num_heads, key_dim=key_dim)
    causal_attention = CausalAttention(num_heads=num_heads, key_dim=key_dim)
    cross_attention = CrossAttention(num_heads=num_heads, key_dim=key_dim)
    self_attention = SelfAttention(num_heads=num_heads, key_dim=key_dim)

    x = tf.random.uniform((3, 100, 128), minval=0, maxval=100, dtype=tf.float32)
    context = tf.random.uniform((3, 100, 128), minval=0, maxval=100, dtype=tf.float32)

    y = base_attention(x)
    assert y.shape == (3, 100, 128)

    y = causal_attention(x)
    assert y.shape == (3, 100, 128)

    y = self_attention(x)
    assert y.shape == (3, 100, 128)

    y = cross_attention(x, context)
    assert y.shape == (3, 100, 128)


def test_ffn():
    ffn = FeedFoward(d_model=128, d_ff=512)

    x = tf.random.uniform((3, 100, 128), minval=0, maxval=100, dtype=tf.float32)

    y = ffn(x)

    assert y.shape == (3, 100, 128)


def test_encoder_decoder_layers():
    encoder_layer = EncoderLayer(d_model=128, num_heads=2, d_ff=512)
    decoder_layer = DecoderLayer(d_model=128, num_heads=2, d_ff=512)

    x = tf.random.uniform((3, 100, 128), minval=0, maxval=100, dtype=tf.float32)
    context = tf.random.uniform((3, 100, 128), minval=0, maxval=100, dtype=tf.float32)

    y = encoder_layer(x)
    assert y.shape == (3, 100, 128)

    y = decoder_layer(x, context)
    assert y.shape == (3, 100, 128)
