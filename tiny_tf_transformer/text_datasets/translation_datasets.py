import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

from typing import Tuple, Union
import re

RESERVED_TOKEN = ["[PAD]", "[UNK]", "[START]", "[END]"]


def write_vocab_to_file(
    text_ds: tf.data.Dataset, vocab_size: int, output_path: str, lower_case: bool = True
):
    """
    Write the wordpiece vocabulary to a file using the text_ds dataset.
    """

    vocab = bert_vocab.bert_vocab_from_dataset(
        text_ds,
        vocab_size=vocab_size,
        reserved_tokens=RESERVED_TOKEN,
        bert_tokenizer_params=dict(lower_case=lower_case),
    )

    tf.io.write_file(output_path, "\n".join(vocab))


def get_wmt19_zh_en_ds() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Returns the wmt19 zh-en translation dataset.
    """

    config = tfds.translate.wmt.WmtConfig(
        version="1.0.1",
        language_pair=("zh", "en"),
        subsets={
            tfds.Split.TRAIN: [
                "newscommentary_v13",
            ],
            tfds.Split.VALIDATION: ["newstest2018"],
        },
    )

    builder = tfds.builder("wmt_translate", config=config)
    builder.download_and_prepare()

    return builder.as_dataset(as_supervised=True)


def write_vocab_zh_en(
    example_pairs: tf.data.Dataset,
    zh_vocab_size: int,
    en_vocab_size: int,
    zh_file: str = "zh_vocab.txt",
    en_file: str = "en_vocab.txt",
):
    """
    Write the wordpiece vocab for zh and en to files using only the training data.
    """

    train_zh = (
        example_pairs.map(lambda zh, en: zh).batch(2048).prefetch(tf.data.AUTOTUNE)
    )
    train_en = (
        example_pairs.map(lambda zh, en: en).batch(2048).prefetch(tf.data.AUTOTUNE)
    )

    write_vocab_to_file(train_zh, zh_vocab_size, zh_file)
    write_vocab_to_file(train_en, en_vocab_size, en_file)


def add_start_end(tokens: Union[tf.RaggedTensor, tf.Tensor]) -> tf.RaggedTensor:
    """
    Add start and end tokens to the ragged tensor.
    If the input is a tensor, it is assumed to be a 1D tensor of strings.
    """

    start_id = tf.constant(RESERVED_TOKEN.index("[START]"), dtype=tf.int64)
    end_id = tf.constant(RESERVED_TOKEN.index("[END]"), dtype=tf.int64)

    if isinstance(tokens, tf.RaggedTensor):
        start = tf.fill([tokens.nrows(), 1], start_id)
        end = tf.fill([tokens.nrows(), 1], end_id)
        return tf.concat([start, tokens, end], axis=1)
    elif isinstance(tokens, tf.Tensor):
        assert tokens.shape.rank == 1  # Only works for 1D tensors
        return tf.concat([[start_id], tokens, [end_id]], axis=0)


def cleanup_text(token_txt: tf.RaggedTensor) -> tf.Tensor:
    """
    Remove the start, end and pad tokens from the tokenized text, while keeping the unk tokens.

    `token_txt` is a ragged tensor of strings, something like:
        <tf.RaggedTensor [[b'[START]', b'hello', b',', b'world', b'!', b'[END]'],
         [b'[START]', b'what', b"'", b's', b'up', b'!', b'[END]']]>o

    Output is a tensor of strings, something like:
        <tf.Tensor: shape=(2,), dtype=string, numpy=array([b'hello , world !', b"what ' s up !"], dtype=object)>
    """

    bad_tokens = [re.escape(t) for t in RESERVED_TOKEN if t != "[UNK]"]

    # Get the bad tokens as a regex, something like: \[PAD\]|\[START\]|\[END\]
    bad_token_re = "|".join(bad_tokens)

    bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
    result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

    result = tf.strings.reduce_join(result, separator=" ", axis=-1)

    return result


class BertTokenizer(tf.Module):
    """
    A custom tokenizer that uses the tensorflow_text BertTokenizer.
    """

    def __init__(
        self,
        vocab_path: str,
        lower_case: bool = True,
        use_fast_bert: bool = False,
        reserved_tokens: list = RESERVED_TOKEN,
    ):
        self._reserved_tokens = reserved_tokens
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = tf.io.read_file(vocab_path)

        self.vocab = tf.strings.split(vocab, "\n")

        self.use_fast_bert = use_fast_bert

        if use_fast_bert:
            raise NotImplementedError("FastBertTokenizer is not implemented yet")
            """
            vocab_np = tf.strings.split(vocab, "\n").numpy()
            vocab_np = [v.decode("utf-8") for v in vocab_np]
            self.tokenizer = tf_text.FastBertTokenizer(vocab_np, support_detokenization=True)# , lower_case=lower_case)
            """
        else:
            self.tokenizer = tf_text.BertTokenizer(vocab_path, lower_case=lower_case)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def tokenize(self, strings: tf.Tensor) -> tf.RaggedTensor:
        """
        Tokenize the strings using the BertTokenizer.
        """

        enc = self.tokenizer.tokenize(strings)
        enc = enc.merge_dims(-2, -1)

        # Add the start and end tokens
        enc = add_start_end(enc)

        return enc

    # add two signatures to the function
    # one for tensor and one for ragged tensor
    @tf.function(
        input_signature=[tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64)]
    )
    def detokenize(self, tokenized: tf.RaggedTensor) -> tf.Tensor:
        """
        Detokenize the tokenized tensor.
        """

        uncleaned_text = self.tokenizer.detokenize(tokenized)
        return cleanup_text(uncleaned_text)

    @tf.function(
        input_signature=[tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64)]
    )
    def lookup(self, token_ids: tf.RaggedTensor) -> tf.RaggedTensor:
        """
        Lookup the token ids in the vocab.
        """

        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self) -> tf.Tensor:
        """
        Return the vocab size.
        """

        return tf.shape(self.vocab)[0]

    @tf.function
    def get_reserved_tokens(self) -> tf.Tensor:
        """
        Return the reserved tokens.
        """

        return self._reserved_tokens


class CharacterTokenizer(tf.Module):
    def __init__(self):
        super(CharacterTokenizer, self).__init__()

        self.char_to_id = tf.constant(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
            ],
            dtype=tf.int32,
        )

        self.char_vocab = tf.constant(
            [
                " ",
                "!",
                "#",
                "$",
                "%",
                "&",
                "'",
                "(",
                ")",
                "*",
                "+",
                ",",
                "-",
                ".",
                "/",
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                ":",
                ";",
                "=",
                "?",
                "@",
                "[",
                "_",
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
                "j",
                "k",
                "l",
                "m",
                "n",
                "o",
                "p",
                "q",
                "r",
                "s",
                "t",
                "u",
                "v",
                "w",
                "x",
                "y",
                "z",
                "~",
            ]
        )

        self.num_chars = tf.shape(self.char_vocab)[0]

        self.reserved_tokens = ["[PAD]", "[START]", "[END]"]

        self.char_vocab = tf.concat([self.char_vocab, self.reserved_tokens], axis=0)
        self.char_to_id = tf.concat(
            [
                self.char_to_id,
                tf.constant(
                    [
                        i
                        for i in range(
                            len(self.char_vocab) - len(self.reserved_tokens),
                            len(self.char_vocab),
                        )
                    ]
                ),
            ],
            axis=0,
        )

        self.table_tokenize = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(self.char_vocab, self.char_to_id),
            default_value=-1,
        )

        self.table_detokenize = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(self.char_to_id, self.char_vocab),
            default_value="",
        )

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def tokenize(self, string: tf.Tensor) -> tf.Tensor:
        """
        Tokenize the string by converting characters to integer IDs.
        """

        num_examples = tf.shape(string)[0]

        chars = tf.strings.unicode_split(string, "UTF-8")
        tokens = self.table_tokenize.lookup(chars)

        # add start and end tokens
        start_id = self.table_tokenize.lookup(tf.convert_to_tensor("[START]"))

        end_id = self.table_tokenize.lookup(tf.convert_to_tensor("[END]"))

        start_token = tf.fill([1], start_id)
        end_token = tf.fill([1], end_id)

        tokens = tf.concat([[start_token], tokens, [end_token]], axis=0)

        # merge the dimensions:
        # i.e. [[start_token], [4, 5, 6], [end_token]] -> [[start_token, 4, 5, 6, end_token]]

        tokens = tf.reshape(tokens, [num_examples, -1])

        return tokens

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
    def detokenize(self, ids: tf.Tensor) -> tf.Tensor:
        """
        Detokenize the IDs by converting them back to characters.

        Input:
            ids: tf.Tensor of shape [batch_size, seq_len]
        """

        # remove [START], [END]
        ids = ids[:, 1:-1]

        char_list = self.table_detokenize.lookup(ids)

        string = tf.strings.reduce_join(char_list, axis=-1)

        return string
