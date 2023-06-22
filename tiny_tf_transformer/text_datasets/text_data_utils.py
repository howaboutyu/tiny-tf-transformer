import tensorflow as tf
import tensorflow_text as text
import tensorflow_datasets as tfds
from typing import Tuple, Union, Mapping
import re


from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

# A list of reserved tokens to add to the vocab
RESERVED_TOKEN = ["[PAD]", "[UNK]", "[START]", "[END]"]


def write_vocab_to_file(
    text_ds: tf.data.Dataset,
    vocab_size: int,
    output_path: str,
    lower_case: bool = True,
    reserved_tokens: list = RESERVED_TOKEN,
):
    """
    Write the wordpiece vocab to a file using only the training data.

    Inputs:
        text_ds: tf.data.Dataset of strings
        vocab_size: int, size of the vocab
        output_path: str, path to write the vocab to
        lower_case: bool, whether to lower case the text
        reserved_tokens: list of strings, reserved tokens to add to the vocab

    Returns:
        None

    """

    vocab = bert_vocab.bert_vocab_from_dataset(
        text_ds,
        vocab_size=vocab_size,
        reserved_tokens=reserved_tokens,
        bert_tokenizer_params=dict(lower_case=lower_case),
    )

    tf.io.write_file(output_path, "\n".join(vocab))


def get_wmt19_zh_en_ds(
    train_split_list: list = ["newscommentary_v13"],
    val_split_list: list = ["newstest2018"],
) -> Mapping[tf.data.Dataset, tf.data.Dataset]:
    """
    Use tfds to get the wmt19 zh-en dataset train and validation splits.

    Inputs:
        train_split_list: list of strings, the train splits to use
        val_split_list: list of strings, the validation splits to use

    Returns:
        translation examples, where examples['train'] is the train split and examples['validation'] is the validation split

    """

    config = tfds.translate.wmt.WmtConfig(
        version="0.0.1",
        language_pair=("zh", "en"),
        subsets={
            tfds.Split.TRAIN: train_split_list,
            tfds.Split.VALIDATION: val_split_list,
        },
    )

    builder = tfds.builder("wmt_translate", config=config)
    builder.download_and_prepare()

    return builder.as_dataset(as_supervised=True)


def add_start_end(
    tokens: Union[tf.RaggedTensor, tf.Tensor],
    reserved_tokens: list = RESERVED_TOKEN,
) -> tf.RaggedTensor:
    """
    Add start and end tokens to the ragged tensor.
    If the input is a tensor, it is assumed to be a 1D tensor of strings.

    Inputs:
        tokens: tf.RaggedTensor or tf.Tensor of strings
        reserved_tokens: list of strings, reserved tokens to add to the vocab

    Returns:
        ragged tensor of strings with start and end tokens added
    """

    start_id = tf.constant(reserved_tokens.index("[START]"), dtype=tf.int64)
    end_id = tf.constant(reserved_tokens.index("[END]"), dtype=tf.int64)

    if isinstance(tokens, tf.RaggedTensor):
        start = tf.fill([tokens.nrows(), 1], start_id)
        end = tf.fill([tokens.nrows(), 1], end_id)
        return tf.concat([start, tokens, end], axis=1)
    elif isinstance(tokens, tf.Tensor):
        assert tokens.shape.rank == 1  # Only works for 1D tensors
        return tf.concat([[start_id], tokens, [end_id]], axis=0)


def cleanup_text(
    token_txt: tf.RaggedTensor,
    reserved_tokens: list = RESERVED_TOKEN,
) -> tf.Tensor:
    """
    Remove the start, end and pad tokens from the tokenized text, while keeping the unk tokens.


    Input:
        token_txt: tf.RaggedTensor of strings
        reserved_tokens: list of strings, reserved tokens to add to the vocab
    Output:
        tf.Tensor of strings, where the start, end and pad tokens have been removed.

    Example:
        The input `token_txt` is a ragged tensor of strings, something like:
        input = <tf.RaggedTensor [[b'[START]', b'hello', b',', b'world', b'!', b'[END]'],
         [b'[START]', b'what', b"'", b's', b'up', b'!', b'[END]']]>o
        >>> cleanup_text(input)
        <tf.Tensor: shape=(2,), dtype=string, numpy=array([b'hello , world !', b"what ' s up !"], dtype=object)>
    """

    bad_tokens = [re.escape(t) for t in reserved_tokens if t != "[UNK]"]

    # Get the bad tokens as a regex, something like: \[PAD\]|\[START\]|\[END\]
    bad_token_re = "|".join(bad_tokens)

    bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
    result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

    result = tf.strings.reduce_join(result, separator=" ", axis=-1)

    return result
