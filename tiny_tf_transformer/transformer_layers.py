""" Building blocks of the transformer model."""

import tensorflow as tf


class BaseAttention(tf.keras.layers.Layer):
    """
    This is the base attention layer.
    It has the layers required to build other attention layers.

    Commonly used attention layers are:
        * self.multi_head_attention is the tf.keras.layers.MultiHeadAttention layer.
        * self.layer_norm is the tf.keras.layers.LayerNormalization layer.
        * self.attention_scores stores the attention scores for visualization for cross attention layers.

    """

    def __init__(self, num_heads: int, key_dim: int, attention_dropout=0.1):
        super(BaseAttention, self).__init__()

        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=attention_dropout
        )

        # Store the attention scores for visualization
        self.attention_scores = None

        # Layer normalization
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)


class CrossAttention(BaseAttention):
    """This layers takens in a context and a query and produces the cross attention."""

    def call(self, x: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        # x is the query, and context is the key and value usually the encoder output
        attention_output, attention_score = self.multi_head_attention(
            query=x, value=context, key=context, return_attention_scores=True
        )

        # Store the attention weights for visualization
        self.attention_scores = attention_score

        x = attention_output + x
        x = self.layer_norm(x)

        return x


class CausalAttention(BaseAttention):
    """This layers uses the causal mask to prevent the decoder from looking ahead."""

    def call(self, x: tf.Tensor) -> tf.Tensor:
        attention_output = self.multi_head_attention(
            query=x, value=x, key=x, use_causal_mask=True
        )

        x = attention_output + x
        x = self.layer_norm(x)

        return x


class SelfAttention(BaseAttention):
    """This layer takes in a query and produces the self attention."""

    def call(self, x: tf.Tensor) -> tf.Tensor:
        attention_output = self.multi_head_attention(
            query=x, value=x, key=x, return_attention_scores=False
        )

        x = attention_output + x
        x = self.layer_norm(x)

        return x


class FeedFoward(tf.keras.layers.Layer):
    """This is the feed forward network for both the encoder and decoder"""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()

        self.dense1 = tf.keras.layers.Dense(d_ff, activation=activation)
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.drop_out = tf.keras.layers.Dropout(dropout_rate)

        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x_original = x

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.drop_out(x)

        # skip connection
        x = x_original + x
        x = self.layer_norm(x)

        return x


class EncoderLayer(tf.keras.layers.Layer):
    """
    This is the encoder layer. Which is made up of:
        1. self_attention
        2. feed_forward
    """

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        d_ff: int,
        attention_dropout: float = 0.1,
        ff_dropout_rate: float = 0.1,
        ff_activation: str = "relu",
    ):
        super().__init__()

        self.self_attention = SelfAttention(
            num_heads, key_dim=d_model, attention_dropout=attention_dropout
        )
        self.ffn = FeedFoward(
            d_model=d_model,
            d_ff=d_ff,
            dropout_rate=ff_dropout_rate,
            activation=ff_activation,
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.self_attention(x)
        x = self.ffn(x)

        return x


class DecoderLayer(tf.keras.layers.Layer):
    """The decoder layer. Which is made up of:
    1. causal_attention
    2. cross_attention
    3. feed_forward
    """

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        attention_dropout_rate=0.1,
        ff_dropout_rate=0.1,
        ff_activation="relu",
    ):
        super(DecoderLayer, self).__init__()

        self.caual_attention = CausalAttention(
            num_heads, d_model, attention_dropout_rate
        )
        self.cross_attention = CrossAttention(
            num_heads, d_model, attention_dropout_rate
        )

        self.ffn = FeedFoward(
            d_model=d_model,
            d_ff=d_ff,
            dropout_rate=ff_dropout_rate,
            activation=ff_activation,
        )

    def call(self, x: tf.Tensor, enc_output: tf.Tensor) -> tf.Tensor:
        x = self.caual_attention(x)
        x = self.cross_attention(x, enc_output)
        x = self.ffn(x)

        return x
