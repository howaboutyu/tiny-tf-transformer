import tensorflow as tf
import tensorflow_text as text

from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab


def write_vocab_to_file(
    text_ds: tf.data.Dataset, vocab_size: int, output_path: str, lower_case: bool = True
):
    vocab = bert_vocab.bert_vocab_from_dataset(
        text_ds,
        vocab_size=vocab_size,
        reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"],
        bert_tokenizer_params=dict(lower_case=lower_case),
    )

    tf.io.write_file(output_path, "\n".join(vocab))
