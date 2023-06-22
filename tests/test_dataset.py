import tensorflow as tf
import tensorflow_text as text
import pytest
import numpy as np

import context

from tiny_tf_transformer.text_datasets.tokenizers import (
    BertTokenizer,
    CharacterTokenizer,
    CharacterTokenizerModel,
)

from tiny_tf_transformer.text_datasets.text_data_utils import (
    add_start_end,
    cleanup_text,
    write_vocab_to_file,
    get_wmt19_zh_en_ds,
)


@pytest.fixture
def en_tokenizer() -> text.BertTokenizer:
    test_ds = tf.data.Dataset.from_tensor_slices(
        ["Hello, world!", "what's up!!?", "how is it going?"]
    )

    write_vocab_to_file(test_ds, 100, "/tmp/test_vocab.txt")

    tokenizer = text.BertTokenizer("/tmp/test_vocab.txt", lower_case=True)

    return tokenizer


@pytest.fixture
def dummy_text_batch() -> tf.data.Dataset:
    test_ds = tf.data.Dataset.from_tensor_slices(
        ["Hello, world!", "what's up!!?", "how is it going?"]
    )

    return test_ds


@pytest.fixture
def dummy_vocab_path() -> str:
    dummy_vocab = """[PAD]
[UNK]
[START]
[END]
he
##llo
,
world
!"""
    with open("/tmp/dummy_vocab.txt", "w") as f:
        f.write(dummy_vocab)

    return "/tmp/dummy_vocab.txt"


def test_write_vocab_to_file(dummy_text_batch: tf.data.Dataset):
    write_vocab_to_file(dummy_text_batch, 100, "/tmp/test_vocab.txt")

    tokenizer = text.BertTokenizer("/tmp/test_vocab.txt", lower_case=True)

    token = tokenizer.tokenize("hello , world !")
    sentence = tokenizer.detokenize(token)

    assert (
        tf.strings.reduce_join(sentence, separator=" ").numpy().decode("utf-8")
        == "hello , world !"
    )


def test_tokenizer_utils(en_tokenizer):
    """
    Test `add_start_end`
    """
    text_rtensor = tf.ragged.constant(
        [
            "Hello, world!",
            "What's up!",
        ]
    )

    token_tensor = en_tokenizer.tokenize(text_rtensor)
    token_tensor = token_tensor.merge_dims(-2, -1)

    token_tensor_with_start_end = add_start_end(token_tensor)

    words = en_tokenizer.detokenize(token_tensor_with_start_end)

    string_out = tf.strings.reduce_join(words, separator=" ", axis=-1)

    assert string_out.numpy()[0].decode("utf-8") == "[START] hello , world ! [END]"
    assert string_out.numpy()[1].decode("utf-8") == "[START] what ' s up ! [END]"
    assert token_tensor.shape == [2, None]

    """
    Test `cleanup_text` 
    """

    text_rtensor = tf.ragged.constant(
        [
            "Hello, @ world!",
            "Wh*at's u#p!",
        ]
    )

    token_tensor = en_tokenizer.tokenize(text_rtensor)
    token_tensor = token_tensor.merge_dims(-2, -1)

    token_tensor_with_start_end = add_start_end(token_tensor)

    words = en_tokenizer.detokenize(token_tensor_with_start_end)

    clean_up = cleanup_text(words)

    assert clean_up.numpy()[0].decode("utf-8") == "hello , [UNK] world !"
    assert clean_up.numpy()[1].decode("utf-8") == "wh [UNK] at ' s u [UNK] p !"


def test_zh_en_wmt19():
    zh_ds = tf.data.Dataset.from_tensor_slices(
        [
            "你好，世界",
            "你好，世界",
        ]
    )

    en_ds = tf.data.Dataset.from_tensor_slices(
        [
            "hello , world !",
            "hello , world !",
        ]
    )

    train_ds = tf.data.Dataset.zip((zh_ds, en_ds))

    write_vocab_to_file(zh_ds, 100, "/tmp/zh_vocab.txt")
    write_vocab_to_file(en_ds, 100, "/tmp/en_vocab.txt")

    zh_tokenizer = text.BertTokenizer("/tmp/zh_vocab.txt", lower_case=True)
    en_tokenizer = text.BertTokenizer("/tmp/en_vocab.txt", lower_case=True)

    zh_token = zh_tokenizer.tokenize("你好，世界")
    zh_sentence = zh_tokenizer.detokenize(zh_token)

    en_token = en_tokenizer.tokenize("hello , world !")
    en_sentence = en_tokenizer.detokenize(en_token)

    assert (
        tf.strings.reduce_join(zh_sentence, separator=" ").numpy().decode("utf-8")
        == "你 好 , 世 界"
    )
    assert (
        tf.strings.reduce_join(en_sentence, separator=" ").numpy().decode("utf-8")
        == "hello , world !"
    )


def test_custom_tokenizer_class(dummy_vocab_path: str):
    en_tokenizer_pre_save = BertTokenizer(dummy_vocab_path, lower_case=True)

    # test save and load using saved model
    tf.saved_model.save(en_tokenizer_pre_save, "/tmp/test_custom_tokenizer_class")
    en_tokenizer = tf.saved_model.load("/tmp/test_custom_tokenizer_class")

    tokens = en_tokenizer.tokenize(["hello , world !"])
    token_lookup = en_tokenizer.lookup(tokens)
    cleaned_text = en_tokenizer.detokenize(tokens)

    assert cleaned_text.numpy()[0].decode("utf-8") == "hello , world !"
    assert token_lookup.numpy()[0][1].decode("utf-8") == "he"
    assert en_tokenizer.get_vocab_size() == 9
    assert np.array_equal(
        en_tokenizer.get_reserved_tokens(), ["[PAD]", "[UNK]", "[START]", "[END]"]
    )


@pytest.mark.skip(reason="Not implemented yet")
def test_custom_fast_bert_tokenizer(dummy_vocab_path: str):
    fast_bert_tokenizer = BertTokenizer(
        dummy_vocab_path, lower_case=True, use_fast_bert=True
    )

    tokens = fast_bert_tokenizer.tokenize(["hello , world !"])
    cleaned_text = fast_bert_tokenizer.detokenize(tokens)

    assert cleaned_text.numpy()[0].decode("utf-8") == "hello , world !"
    assert tokens.numpy()[0][1].decode("utf-8") == "he"
    assert fast_bert_tokenizer.get_vocab_size() == 9


def test_character_tokenizer():
    char_tokenizer = CharacterTokenizer()

    tokens = char_tokenizer.tokenize(tf.constant(["hello , world ! 1+1=2"]))

    characters = char_tokenizer.detokenize(tokens)

    assert characters.numpy()[0].decode("utf-8") == "hello , world ! 1+1=2"

    char_model = CharacterTokenizerModel(char_tokenizer)

    char_model(tf.constant(["hello , world ! 1+1=2"]))
    char_model.save("/tmp/test_character_tokenizer")

    #################
    #  Test tflite  #
    #################
    converter = tf.lite.TFLiteConverter.from_saved_model(
        "/tmp/test_character_tokenizer"
    )
    converter.allow_custom_ops = True
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    tflite_model = converter.convert()

    # save tflite model
    with open("/tmp/test_character_tokenizer.tflite", "wb") as f:
        f.write(tflite_model)

    # load tflite model
    interpreter = tf.lite.Interpreter(model_path="/tmp/test_character_tokenizer.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(
        input_details[0]["index"], np.array(["hello , world ! 1+1=2"])
    )

    interpreter.invoke()

    tokens_tflite = interpreter.get_tensor(output_details[0]["index"])

    # check tflite tokens are the same as the original tokens
    assert np.array_equal(tokens.numpy(), tokens_tflite)
