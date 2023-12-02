import tensorflow as tf
from tiny_tf_transformer.transformer import Decoder, Encoder
from tiny_tf_transformer.embedding_layers import PositionalEmbedding


def get_target_vocab_size(
    num_bins,
):
    # 3 special tokens: <pad>, <sos>, <eos>
    # and 1 for shifting the values by 1
    return num_bins + 4


def get_model_and_preprocessing(model_name):
    model_preprocessing_dict = {
        "resnet50v2": {
            "model": tf.keras.applications.ResNet50V2,
            "preprocess": tf.keras.applications.resnet_v2.preprocess_input,
        },
        "mobilenetv2": {
            "model": tf.keras.applications.MobileNetV2,
            "preprocess": tf.keras.applications.mobilenet_v2.preprocess_input,
        },
    }

    model_info = model_preprocessing_dict.get(model_name)

    if model_info:
        kwargs = {"include_top": False, "weights": "imagenet"}
        return {
            "model": model_info["model"](**kwargs),
            "preprocess": model_info["preprocess"],
        }
    else:
        return None


def get_pix2seq_model(
    input_shape,
    model_name="resnet50v2",
    num_layers: int = 4,
    d_model: int = 128,
    num_heads: int = 8,
    d_ff: int = 512,
    target_vocab_size: int = 256,
    attention_dropout_rate: float = 0.1,
    ff_dropout_rate: float = 0.1,
    max_length: int = 2048,
):
    feature_extraction_info = get_model_and_preprocessing(model_name)
    preprocessor_fn = feature_extraction_info["preprocess"]
    feature_extractor = feature_extraction_info["model"]

    # encoder
    position_embedding = PositionalEmbedding(d_model=d_model, max_length=max_length)
    encoder = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        vocab_size=target_vocab_size,
        num_heads=num_heads,
        d_ff=d_ff,
        attention_dropout_rate=attention_dropout_rate,
        ff_dropout_rate=ff_dropout_rate,
        pos_embedding_fn=position_embedding,
    )

    # get decoder model
    decoder = Decoder(
        num_layers=num_layers,
        d_model=d_model,
        vocab_size=target_vocab_size,
        num_heads=num_heads,
        d_ff=d_ff,
        attention_dropout_rate=attention_dropout_rate,
        ff_dropout_rate=ff_dropout_rate,
        max_length=max_length,
    )

    input = tf.keras.layers.Input(shape=input_shape)
    decoder_input = tf.keras.layers.Input(shape=(max_length,))

    x = preprocessor_fn(input)

    x = feature_extractor(x)

    # reshape feature grid from [n, n, d] to [n * n, d]
    x = tf.keras.layers.Reshape((-1, x.shape[-1]))(x)

    x = encoder(x)

    # use a causal decoder
    x = decoder(decoder_input, x)

    x = tf.keras.layers.Dense(target_vocab_size, name="final_layer")(x)

    model = tf.keras.Model(inputs=[input, decoder_input], outputs=x)

    print(model.summary())
    return model


class WarmupThenDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, epochs, warmup_epochs, steps_per_epoch):
        super(WarmupThenDecaySchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.steps_per_epoch = steps_per_epoch

    def __call__(self, step):
        warmup_lr = self.initial_learning_rate * (
            step / (self.warmup_epochs * self.steps_per_epoch)
        )
        decay_lr = self.initial_learning_rate * (
            1 - step / (self.epochs * self.steps_per_epoch)
        )
        return tf.minimum(warmup_lr, decay_lr)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "epochs": self.epochs,
            "warmup_epochs": self.warmup_epochs,
            "steps_per_epoch": self.steps_per_epoch,
        }
