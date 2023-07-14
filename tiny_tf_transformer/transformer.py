""" This is the transformer implementation. It is made up of:
    1. Encoder
    2. Decoder
    3. Transformer
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional

from .embedding_layers import PositionalTokenEmbedding
from .transformer_layers import EncoderLayer, DecoderLayer


class Encoder(tf.keras.layers.Layer):
    """
    The encoder is made up of:
    Positional Encoding -> Encoder Layers defined by `num_layers`

    The positional encoding/embedding can be passed in as a function, if not
    then the default sine cosine positional encoding with token embedding is used.


    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        vocab_size: int,
        num_heads: int,
        d_ff: int,
        attention_dropout_rate: float = 0.1,
        ff_dropout_rate: float = 0.1,
        max_length: int = 2048,
        pos_embedding_fn: Optional[tf.keras.layers.Layer] = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        if pos_embedding_fn is None:
            self.pos_embedding = PositionalTokenEmbedding(
                vocab_size=vocab_size, d_model=d_model, max_length=max_length
            )
        else:
            self.pos_embedding = pos_embedding_fn

        # create a list of encoder layers
        self.encoder_layers = [
            EncoderLayer(
                num_heads=num_heads,
                d_model=d_model,
                d_ff=d_ff,
                attention_dropout=attention_dropout_rate,
                ff_dropout_rate=ff_dropout_rate,
            )
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(ff_dropout_rate)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.pos_embedding(x)
        
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.encoder_layers[i](x)

        return x


class Decoder(tf.keras.layers.Layer):
    """

    Decoder layer with the following structure:

    Positional Encoding -> Decoder Layers

        * The positional encoding/embedding can be passed in as a function, if not
          then the default sine cosine positional encoding with token embedding is used.

        * The decoder layer uses causal attention with cross attention using context
          information from the encoder network.




    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        attention_dropout_rate: float = 0.1,
        ff_dropout_rate: float = 0.1,
        max_length: int = 2048,
        pos_embedding_fn: Optional[tf.keras.layers.Layer] = None,
    ):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        if pos_embedding_fn is None:
            self.pos_embedding = PositionalTokenEmbedding(
                vocab_size=vocab_size, d_model=d_model, max_length=max_length
            )
        else:
            self.pos_embedding = pos_embedding_fn

        self.decoder_layers = [
            DecoderLayer(
                num_heads=num_heads,
                d_model=d_model,
                d_ff=d_ff,
                attention_dropout_rate=attention_dropout_rate,
                ff_dropout_rate=ff_dropout_rate,
            )
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(ff_dropout_rate)

    def call(self, x: tf.Tensor, enc_output: tf.Tensor) -> tf.Tensor:
        x = self.pos_embedding(x)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.decoder_layers[i](x, enc_output)

        return x


class Transformer(tf.keras.Model):
    """
    The transformer model is made up of:
    1. Encoder
    2. Decoder
    3. Final linear layer to map the output to the vocabulary size.

    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        input_vocab_size: int,
        target_vocab_size: int,
        attention_dropout_rate: float = 0.1,
        ff_dropout_rate: float = 0.1,
        max_length: int = 2048,
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            vocab_size=input_vocab_size,
            num_heads=num_heads,
            d_ff=d_ff,
            attention_dropout_rate=attention_dropout_rate,
            ff_dropout_rate=ff_dropout_rate,
            max_length=max_length,
        )

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            vocab_size=target_vocab_size,
            num_heads=num_heads,
            d_ff=d_ff,
            attention_dropout_rate=attention_dropout_rate,
            ff_dropout_rate=ff_dropout_rate,
            max_length=max_length,
        )

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(
        self, inputs: Tuple[tf.Tensor, tf.Tensor], training: Optional[bool] = None
    ) -> tf.Tensor:
        # Shape of input and target:
        # input.shape = (batch_size, input_seq_len)
        # target.shape = (batch_size, target_seq_len)
        input, target = inputs

        # Shape of enc_output and dec_output:
        # enc_output.shape = (batch_size, input_seq_len, d_model)
        # dec_output.shape = (batch_size, target_seq_len, d_model)
        enc_output = self.encoder(input)
        dec_output = self.decoder(target, enc_output)

        final_output = self.final_layer(dec_output)

        return final_output
