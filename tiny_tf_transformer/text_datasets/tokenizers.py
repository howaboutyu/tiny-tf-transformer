import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

from tiny_tf_transformer.text_datasets.text_data_utils import (
    RESERVED_TOKEN,
    cleanup_text,
    add_start_end,
)

DEFAULT_CHAR_VOCAB = [
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

    @tf.function
    def get_start_token_id(self, start_token="[START]") -> tf.Tensor:
        """
        Return the start token id.
        """

        return tf.where(self.vocab == start_token)[0][0]

    @tf.function
    def get_end_token_id(self, end_token="[END]") -> tf.Tensor:
        """
        Return the end token id.
        """

        return tf.where(self.vocab == end_token)[0][0]


class CharacterTokenizer(tf.Module):
    def __init__(self, char_vocab: list[str] = DEFAULT_CHAR_VOCAB):
        super(CharacterTokenizer, self).__init__()

        self.char_vocab = tf.constant(char_vocab)

        self.reserved_tokens = ["[PAD]", "[START]", "[END]"]
        self.char_to_id = tf.range(
            1, len(self.char_vocab) + 1 + len(self.reserved_tokens)
        )

        self.char_vocab = tf.concat([self.char_vocab, self.reserved_tokens], axis=0)

        self.table_tokenize = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(self.char_vocab, self.char_to_id),
            default_value=-1,
        )

        self.table_detokenize = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(self.char_to_id, self.char_vocab),
            default_value="",
        )

    @property
    def vocab_size(self) -> tf.Tensor:
        """
        Return the vocab size.
        """
        return tf.shape(self.char_vocab)[0]

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
    def look_up(self, token_ids: tf.Tensor) -> tf.Tensor:
        """
        Lookup the token ids in the vocab.
        """

        return self.table_detokenize.lookup(token_ids)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def tokenize(self, string: tf.Tensor) -> tf.Tensor:
        """
        Tokenize the string by converting characters to integer IDs.
        """

        num_examples = tf.shape(string)[0]

        chars = tf.strings.unicode_split(string, "UTF-8")
        tokens = self.table_tokenize.lookup(chars)

        # add start and end tokens
        start_id = self.get_start_token_id()
        end_id = self.get_end_token_id()

        start_token = tf.fill([1], start_id)
        end_token = tf.fill([1], end_id)

        tokens = tf.concat([[start_token], tokens, [end_token]], axis=0)

        # merge the dimensions:
        # i.e. [[start_token], [4, 5, 6], [end_token]] -> [[start_token, 4, 5, 6, end_token]]
        tokens = tf.reshape(tokens, [num_examples, -1])

        return tokens

    # @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
    def detokenize(self, ids: tf.Tensor) -> tf.Tensor:
        """
        Detokenize the IDs by converting them back to characters.

        Input:
            ids: tf.Tensor of shape [batch_size, seq_len]
        """

        ids = ids[:, 1:-1]
        char_list = self.table_detokenize.lookup(ids)
        # import pdb; pdb.set_trace()

        # string = cleanup_text(char_list)
        string = tf.strings.reduce_join(char_list, axis=-1)

        return string

    def get_start_token_id(self, start_token="[START]") -> tf.Tensor:
        """
        Return the start token id.
        """
        start_token_id = self.table_tokenize.lookup(tf.convert_to_tensor(start_token))
        return start_token_id

    def get_end_token_id(self, end_token="[END]") -> tf.Tensor:
        """
        Return the end token id.
        """
        end_token_id = self.table_tokenize.lookup(tf.convert_to_tensor(end_token))
        return end_token_id

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
