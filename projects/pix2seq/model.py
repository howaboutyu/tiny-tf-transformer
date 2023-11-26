import tensorflow as tf
from tiny_tf_transformer.transformer import Decoder


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
    if model_name == "resnet50v2":
        feature_extractor = tf.keras.applications.ResNet50V2(
            include_top=False, weights="imagenet"
        )
        preprocess_fn = tf.keras.applications.resnet_v2.preprocess_input

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

    x = preprocess_fn(input)

    # extract features
    x = feature_extractor(x)

    # reshape context from [n, n, d] to [n * n, d]
    x = tf.keras.layers.Reshape((-1, x.shape[-1]))(x)

    # reduce dimensionality
    #x = tf.keras.layers.Dense(d_ff, activation='relu')(x)

    # use a causal decoder
    x = decoder(decoder_input, x)

    x = tf.keras.layers.Dense(target_vocab_size, name="final_layer")(x)

    model = tf.keras.Model(inputs=[input, decoder_input], outputs=x)

    print(model.summary())
    return model
