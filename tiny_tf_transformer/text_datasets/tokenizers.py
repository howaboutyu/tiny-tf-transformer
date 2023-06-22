import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

from tiny_tf_transformer.text_datasets.text_data_utils import (
    RESERVED_TOKEN,
    cleanup_text,
    add_start_end,
)


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

    # the call function
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def __call__(self, string: tf.Tensor) -> tf.Tensor:
        """
        Tokenize the string by converting characters to integer IDs.
        """

        return self.tokenize(string)


class CharacterTokenizerModel(tf.keras.Model):
    """
    A Keras model that tokenizes the inputs using the CharacterTokenizer.
    This can be saved and converted to tflite
    """

    def __init__(self, tokenizer: CharacterTokenizer):
        super(CharacterTokenizerModel, self).__init__()

        self.tokenizer = tokenizer

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Tokenize the inputs.
        """

        return self.tokenizer(inputs)
