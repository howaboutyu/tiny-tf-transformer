""" Layers related to embedding. """

import tensorflow as tf
import numpy as np


class PositionalTokenEmbedding(tf.keras.layers.Layer):
    """
    Given a token look up its token embedding and also adds a positional embedding.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_length: int,
        output_dtype: tf.DType = tf.float32,
    ):
        super(PositionalTokenEmbedding, self).__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        # max_length is the maximum length of the sequence
        self.max_length = max_length

        # token embedding function
        self.token_embedding = tf.keras.layers.Embedding(
            vocab_size, d_model, mask_zero=True
        )

        # positional encoding vector
        self.pos_encoding = self.positional_encoding(max_length, d_model, output_dtype)

        self.output_dtype = output_dtype

    def compute_mask(self, *args, **kwargs):
        # Override this method to set the correct mask for the token embedding layer
        return self.token_embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        # Multiply by sqrt(d_model) as outlined in section 3.4 of the original transformer paper
        x = self.token_embedding(x)
        x = x * tf.math.sqrt(
            tf.cast(self.d_model, self.output_dtype)
        )  # x.shape = (batch_size, length, d_model)

        # Add the positional encoding
        x = x + self.pos_encoding[:, :length]

        return x

    @staticmethod
    def positional_encoding(length, depth, output_dtype=tf.float32):
        """
        This is the positional encoding function implementation for the original transformer paper.

        Where pos is the position index and i is the dimension index.

        The equations are as follows:

        PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos/10000^(2i/d_model)

        """

        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :]

        angles = positions / np.power(10000, (2 * (depths // 2)) / np.float32(depth))

        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])

        pos_encoding = angles[np.newaxis, ...]

        return tf.cast(pos_encoding, output_dtype)
