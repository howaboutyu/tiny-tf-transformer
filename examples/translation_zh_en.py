"""
An example of translation from Chinese to English using the WMT19 dataset.
"""


import os
import time
from typing import Tuple

from absl import app, flags, logging

import tensorflow as tf
import tensorflow_text as tf_text

from tiny_tf_transformer.text_datasets.text_data_utils import (
    get_wmt19_zh_en_ds,
    write_vocab_to_file,
)

from tiny_tf_transformer.text_datasets.tokenizers import BertTokenizer


from tiny_tf_transformer.transformer import Encoder, Decoder, Transformer
from tiny_tf_transformer.losses_and_metrics import (
    masked_sparse_categorical_accuracy,
    masked_sparse_categorical_cross_entropy,
)

FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 64, "Batch size.")
flags.DEFINE_integer("num_epochs", 100, "Number of epochs.")
flags.DEFINE_integer("num_layers", 4, "Number of layers.")
flags.DEFINE_integer("d_model", 128, "Model dimension.")
flags.DEFINE_integer("d_ff", 512, "Feedforward dimension.")
flags.DEFINE_integer("num_heads", 8, "Number of attention heads.")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate.")
flags.DEFINE_integer("max_token_length", 64, "Maximum token length.")

def main(argv):
    batch_size = FLAGS.batch_size
    num_epochs = FLAGS.num_epochs
    en_vocab_size = 10_000
    zh_vocab_size = 10_000
    num_layers = FLAGS.num_layers
    d_model = FLAGS.d_model
    d_ff = FLAGS.d_ff
    num_heads = FLAGS.num_heads
    learning_rate = FLAGS.learning_rate
    max_token_length = FLAGS.max_token_length
    

    zh_en_examples = get_wmt19_zh_en_ds()

    train_ds = zh_en_examples["train"]
    val_ds = zh_en_examples["validation"]

    for zh, en in train_ds.take(1000):
        print("zh: ", zh.numpy().decode("utf-8"))
        print("en: ", en.numpy().decode("utf-8"))

    # Write the vocab files if they don't exist.
    zh_vocab_file = "zh_vocab.txt"
    en_vocab_file = "en_vocab.txt"

    if not os.path.exists(zh_vocab_file):
        write_vocab_to_file(train_ds.map(lambda zh, en: zh), zh_vocab_size, zh_vocab_file)

    if not os.path.exists(en_vocab_file):
        write_vocab_to_file(train_ds.map(lambda zh, en: en), en_vocab_size, en_vocab_file)

    # Create the tokenizers.
    zh_tokenizer = BertTokenizer(zh_vocab_file)
    en_tokenizer = BertTokenizer(en_vocab_file)

    def prepare_batch(
        zh: tf.Tensor,
        en: tf.Tensor,
        max_token_length: int = max_token_length,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Prepare the batch for the transformer model.

        Inputs:
            zh: tf.Tensor, the Chinese text
            en: tf.Tensor, the English text
            max_token_length: int, the maximum token length to use
        Returns:
            zh: tf.Tensor, the Chinese text
            en_input: tf.Tensor, the English text with the start token prepended
            en_target: tf.Tensor, the English text with the end token appended

        """
        zh = zh_tokenizer.tokenize(zh)
        zh = zh[..., :max_token_length]

        en = en_tokenizer.tokenize(en)
        en = en[..., : (max_token_length + 1)]

        en_input = en[:, :-1]
        en_target = en[:, 1:]

        # from ragged tensor to dense tensor
        zh = zh.to_tensor()
        en_input = en_input.to_tensor()
        en_target = en_target.to_tensor()

        return (zh, en_input), en_target

    train_ds = train_ds.shuffle(10_000)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.map(prepare_batch, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)


    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.map(prepare_batch, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # create the transformer

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        input_vocab_size=zh_tokenizer.get_vocab_size().numpy(),
        target_vocab_size=en_tokenizer.get_vocab_size().numpy(),
        ff_dropout_rate=0.1,
        attention_dropout_rate=0.1,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
    )

    transformer.compile(
        loss=masked_sparse_categorical_cross_entropy,
        optimizer=optimizer,
        metrics=[masked_sparse_categorical_accuracy],
    )

    transformer.fit(
        train_ds,
        epochs=num_epochs,
        validation_data=val_ds,
    )

    # save the model
    transformer.save("transformer_zh_en")


if __name__ == "__main__":
    app.run(main)
