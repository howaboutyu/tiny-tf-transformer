import tensorflow as tf
import os
from tensorflow.keras import mixed_precision
from tensorflow.keras import layers 
import numpy as np

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)

from tiny_tf_transformer.embedding_layers import (
    PositionalTokenEmbedding,
    PositionalEmbedding,
)

from tiny_tf_transformer.transformer_layers import (
    BaseAttention,
    CausalAttention,
    CrossAttention,
    SelfAttention,
    FeedFoward,
    EncoderLayer,
    DecoderLayer,
)


import tensorflow as tf
class CustomConvModel(tf.keras.Model):
    def __init__(self, d_model=128, num_classes=10):
        super(CustomConvModel, self).__init__()

        self.input_layer = layers.InputLayer(input_shape=(None, None, d_model))

        # Branch 1: 1x1 Convolution
        self.branch1 = layers.Conv2D(d_model, (1, 1), activation='relu', padding='same')

        # Branch 2: 3x3 Convolution
        self.branch2 = layers.Conv2D(d_model, (3, 3), activation='relu', padding='same')

        # Branch 3: 5x5 Convolution
        self.branch3 = layers.Conv2D(d_model, (5, 5), activation='relu', padding='same')

        # Additional Convolutional layers after merging
        self.conv1 = layers.Conv2D(64, (2, 1), activation='relu', padding='valid')
        self.conv2 = layers.Conv2D(64, (1, 2), activation='relu', padding='valid')

        self.conv_list = [
            layers.Conv2D(64, (3, 3), activation='relu', padding='valid', groups=4)

            for _ in range(20)
            ]


        #self.pool1 = layers.MaxPooling2D((2, 2))
        self.conv3 = layers.Conv2D(d_model, (3, 3), activation='relu', padding='valid')
        #self.pool2 = layers.MaxPooling2D((2, 2))

        

    def call(self, x):

        # Branches
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)
        branch3_output = self.branch3(x)

        # Merge branches
        merged = layers.concatenate([branch1_output, branch2_output, branch3_output], axis=-1)

        # Additional Convolutional layers after merging
        x = self.conv1(merged)

        for i  in range(8):
            x = self.conv_list[i](x)
        
        
        return x 

class PositionEmbeddingModel(tf.keras.Model):
    def __init__(self, max_height=30, max_width=30, d_model=128, use_conv=True):
        super(PositionEmbeddingModel, self).__init__()
        self.max_height = max_height
        self.max_width = max_width
        self.d_model = d_model
        self.feature_model = CustomConvModel(d_model=d_model)

        self.position_embedding_x_fn = tf.keras.layers.Embedding(
            input_dim=60, output_dim=d_model//2
        )
        self.position_embedding_y_fn = tf.keras.layers.Embedding(
            input_dim=60, output_dim=d_model//2
        )
        self.class_embedding_fn = tf.keras.layers.Embedding(
            input_dim=11, output_dim=d_model
        )

        self.dense = tf.keras.layers.Dense(d_model, activation='relu', use_bias=True)
        self.dense_position = tf.keras.layers.Dense(d_model, activation='relu', use_bias=True)
        self.conv = tf.keras.layers.Conv2D(
            d_model,
            (2, 2),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            activation='relu'
        )

        self.conv2 = tf.keras.layers.Conv2D(
            d_model,
            (2, 2),
            strides=(1, 1),
            padding="same",
            activation='relu',
            use_bias=True,
        )

        self.use_conv = use_conv

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.int8)

        # print(inputs.shape)
        #x = tf.one_hot(inputs, 10, axis=-1)
        x = self.class_embedding_fn(inputs)
        
        #x = tf.nn.relu(x)
        #if self.use_conv:
            #x = self.dense(x)
        #x = self.conv(x)
        #else:
        #    x = self.dense(x)

        w = x.shape[1]
        h = x.shape[2]

        position_embedding_x = self.position_embedding_x_fn(tf.range(self.max_width))
        position_embedding_y = self.position_embedding_y_fn(tf.range(self.max_height))

        position_embedding_x = tf.expand_dims(
            position_embedding_x, 0
        )  # Shape: (max_height, 1, d_model)

        position_embedding_x = tf.repeat(position_embedding_x, self.max_height, 0)
        position_embedding_y = tf.expand_dims(
            position_embedding_y, 1
        )  # Shape: (1, max_width, d_model)
        position_embedding_y = tf.repeat(position_embedding_y, self.max_width, 1)
        pos_x = tf.concat([position_embedding_x  , position_embedding_y], -1)
        pos_x = pos_x[tf.newaxis, ...]

        import pdb; pdb.set_trace()
        x = pos_x + x
        #x = self.feature_model(x)
        num_elements = x.shape[1] * x.shape[2]

        x = tf.reshape(x, (-1, num_elements, self.d_model))

        return x


class Decoder(tf.keras.layers.Layer):
    """ """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attention_dropout_rate: float = 0.1,
        ff_dropout_rate: float = 0.1,
    ):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.decoder_layers = [
            DecoderLayer(
                num_heads=num_heads,
                d_model=d_model,
                d_ff=d_ff,
                attention_dropout_rate=attention_dropout_rate,
                ff_dropout_rate=ff_dropout_rate,
                use_causal=False
            )
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(ff_dropout_rate)

    def call(self, x: tf.Tensor, enc_output: tf.Tensor=None) -> tf.Tensor:
        for i in range(self.num_layers):
            if enc_output is not None:
                x = self.decoder_layers[i](x, enc_output)

            else:
                x = self.decoder_layers[i](x, x)

        return x


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
        num_heads: int,
        d_ff: int,
        attention_dropout_rate: float = 0.1,
        ff_dropout_rate: float = 0.1,
        image_height: int = 30,
        image_width: int = 30,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Instantiate the model
        self.pos_embedding = PositionEmbeddingModel(
            max_height=image_height, max_width=image_width, d_model=d_model
        )

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

    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        x = self.pos_embedding(x)
        
        x_out = []
        for i in range(self.num_layers):
            x = self.encoder_layers[i](x)

            x_out.append(x[:, -1])
        #x_concat = tf.transpose(tf.stack(x_out), (1, 0, 2)) 
        return x 


class ImageTransformerModel(tf.keras.Model):
    def __init__(self, input_shape, target_shape, d_model, d_ff, num_heads, key_dim,
                 attention_dropout, num_attention_heads, num_layers, max_height, max_width):
        super(ImageTransformerModel, self).__init__()
        
        self.input_shape = input_shape
        self.target_shape = target_shape
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attention_dropout = attention_dropout
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.max_height = max_height
        self.max_width = max_width
        self.num_layers_decoder = self.num_layers #* 2

        self.encoder_block = self._build_encoder()
        self.decoder = self._build_decoder()
        self.pos_embedding = PositionEmbeddingModel(
            max_height=target_shape[0], max_width=target_shape[1], d_model=d_model, use_conv=False
        )
        self.dense1 = tf.keras.layers.Dense(32, activation="relu")
        self.dense2 = tf.keras.layers.Dense(10, activation="linear")
        self.avgpool1d = tf.keras.layers.GlobalAveragePooling1D()
        self.dense8x8 =  tf.keras.layers.Dense(8*8*self.d_model, activation="relu")
        self.dense4x4 =  tf.keras.layers.Dense(4*4*self.d_model, activation="relu")
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        self.bn4 = layers.BatchNormalization()
        self.bn5 = layers.BatchNormalization()
        self.bn6 = layers.BatchNormalization()

        self.upsample1 = layers.Conv2DTranspose(d_model, (3, 3), strides=(2, 2), padding='valid', activation='relu')
        self.upsample2 = layers.Conv2DTranspose(d_model, (3, 3), strides=(2, 2), padding='valid', activation='relu')
        self.upsample3 = layers.Conv2DTranspose(d_model, (3, 3), strides=(1, 1), padding='same', activation='relu')
        self.upsample4 = layers.Conv2DTranspose(d_model, (2, 2), strides=(1, 1), padding='valid', activation='relu')
        self.upsample5 = layers.Conv2DTranspose(10, (3, 3), strides=(1, 1), padding='same', activation='linear')



    def _build_decoder(self):
        return Decoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            attention_dropout_rate=self.attention_dropout,
            ff_dropout_rate=0.1,
        )

    def _build_encoder(self):
        return Encoder(
            num_layers=self.num_layers*2,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            attention_dropout_rate=self.attention_dropout,
            ff_dropout_rate=0.1,
            image_height=self.input_shape[0] //2,
            image_width=self.input_shape[1] //2,
        )

    def call(self, inputs):
        # if isinstance(inputs, dict):
        #     inputs_encoder = inputs['input']
        #     inputs_decoder = inputs['decoder_input']
        # else:
        #     inputs_encoder = inputs[0]
        #     inputs_decoder = inputs[1]

        x = self.pos_embedding(inputs[0])

        x = self.decoder(x)

        x = tf.keras.layers.Reshape((self.target_shape[1],self.target_shape[0], self.d_model))(x)

        x = self.dense2(x)

        return x


    def _call(self, inputs):
        if isinstance(inputs, dict):
            inputs_encoder = inputs['input']
            inputs_decoder = inputs['decoder_input']
        else:
            inputs_encoder = inputs[0]
            inputs_decoder = inputs[1]

        # Encoder

        #if 0:
        encoder_features = self.encoder_block(inputs_encoder)

        # Decoder
        x_dec = self.pos_embedding(inputs_decoder)

        encoder_features = self.avgpool1d(encoder_features)
        encoder_features = tf.keras.layers.Reshape(( 1, self.d_model) )(encoder_features)

        #x = encoder_features[:, -20:, -20:]

        #decoder_input = self.encoder_block(inputs_decoder)
        
        decoder_out  = self.decoder(x_dec, encoder_features)
        #x = decoder_out 
        ## Reshape to [batch, height, width, depth]
        #latent_shape = int(np.sqrt(x.shape[1]))
        x = tf.keras.layers.Flatten()(encoder_features)
        x = self.dense4x4(x)

        #x = self.avgpool1d(encoder_features)
        #x = self.dense4x4(x)
        x = tf.keras.layers.Reshape((4,4, self.d_model))(x)
        x = self.upsample1(x)
        #x = tf.keras.layers.Reshape((20,20, self.d_model))(decoder_out)
        x = self.bn1(x)
        x = self.upsample2(x)
        x = self.bn2(x)
        x = self.upsample3(x)
        x = self.bn3(x)
        x = self.upsample4(x)
        x = self.bn4(x)
        #x = self.upsample5(x)

        x = self.dense1(x)
        x = self.dense2(x)

        return x

    def compile_model(self, initial_learning_rate=1e-4, decay_steps=10000, decay_rate=0.99):
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate, decay_steps=decay_steps, decay_rate=decay_rate
        )
        opt = tf.keras.optimizers.AdamW(learning_rate=2e-5) # lr_schedule)
        self.compile(optimizer=opt, loss=loss, run_eagerly=False)

    def summary(self):
        inputs_encoder = tf.keras.layers.Input(shape=self.input_shape, name="input")
        inputs_decoder = tf.keras.layers.Input(shape=self.target_shape, name="decoder_input")
        input_dict = {
                'input': inputs_encoder,
                'input_decoder': inputs_decoder
                }
        #model = tf.keras.Model(inputs=[inputs_encoder, inputs_decoder], outputs=self.call([inputs_encoder, inputs_decoder]))
        model = tf.keras.Model(inputs=[ inputs_decoder], outputs=self.call([ inputs_decoder]))
        model.summary()



# Usage
input_shape = (20 * 4, 20 * 2)
target_height = 20
target_width = 20
max_height = 20 * 4
max_width = 20 * 2
d_model =64
d_ff = 64
num_heads = 8
key_dim = 16
attention_dropout = 0.1
num_attention_heads = 8
num_layers =2 

transformer_model = ImageTransformerModel(
    input_shape=input_shape,
    target_shape=(target_height, target_width),
    d_model=d_model,
    d_ff=d_ff,
    num_heads=num_heads,
    key_dim=key_dim,
    attention_dropout=attention_dropout,
    num_attention_heads=num_attention_heads,
    num_layers=num_layers,
    max_height=max_height,
    max_width=max_width
)

transformer_model.compile_model()
transformer_model.summary()

keras_model = transformer_model

# Load and preprocess data from folders
def load_image(image_path, flip=False):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)

    ##image = tf.cast(image, tf.float32)
    # if flip:
    #    image = tf.image.flip_left_right(image)
    # image = tf.image.resize(image, [30, 30])
    # image = image / 255.0
    return image


def load_data(source_folder, target_source_folder, target_folder):
    source_images = sorted(
        [
            os.path.join(source_folder, f)
            for f in os.listdir(source_folder)
            if f.endswith(".png")
        ]
    )
    target_images = sorted(
        [
            os.path.join(target_folder, f)
            for f in os.listdir(target_folder)
            if f.endswith(".png")
        ]
    )

    target_source_images = sorted(
        [
            os.path.join(target_source_folder, f)
            for f in os.listdir(target_source_folder)
            if f.endswith(".png")
        ]
    )

    print(f"number images: {len(source_images)}")

    target_map = lambda x: load_image(x)
    source_dataset = tf.data.Dataset.from_tensor_slices(source_images).map(
        load_image, num_parallel_calls=tf.data.AUTOTUNE
    )
    target_dataset = tf.data.Dataset.from_tensor_slices(target_images).map(
        target_map, num_parallel_calls=tf.data.AUTOTUNE
    )
    target_source_dataset = tf.data.Dataset.from_tensor_slices(
        target_source_images
    ).map(target_map, num_parallel_calls=tf.data.AUTOTUNE)

    input_ds = tf.data.Dataset.zip((source_dataset, target_source_dataset))

    #dataset = tf.data.Dataset.zip((input_ds, target_dataset))
    dataset = tf.data.Dataset.zip((target_source_dataset, target_source_dataset))

    def generator(inputs, output):
        return {"input": inputs[0], "decoder_input": inputs[1]}, output

    dataset = dataset.map(generator)

    dataset = (
        dataset.shuffle(buffer_size=1024)
        .batch(128)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    
    return dataset


source_folder = "/root/arc/largest"
target_folder = "/root/arc/largest_target"
target_source_folder = "/root/arc/largest_source"

val_source_folder = "/root/arc/largest_val"
val_target_folder = "/root/arc/largest_val_target"
val_target_source_folder = "/root/arc/largest_val_source"



# Load the dataset
train_dataset = load_data(source_folder, target_source_folder, target_folder)
val_dataset = load_data(val_source_folder, val_target_source_folder, val_target_folder)


# Train the model
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#keras_model.load_weights('saved_model.h5')


keras_model.fit(
    train_dataset,
    epochs=100,
    validation_data=val_dataset,
    callbacks=[],
)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the color map
color_map = {
    0: "#000000",  # black
    1: "#0074D9",  # blue
    2: "#FF4136",  # red
    3: "#2ECC40",  # green
    4: "#FFDC00",  # yellow
    5: "#AAAAAA",  # grey
    6: "#F012BE",  # fuchsia
    7: "#FF851B",  # orange
    8: "#7FDBFF",  # teal
    9: "#870C25",  # brown
}

# Convert the color map to a format suitable for use with matplotlib
color_map_rgb = {
    k: tuple(int(color_map[k][i : i + 2], 16) for i in (1, 3, 5)) for k in color_map
}


# Function to map predictions to colors
def map_predictions_to_colors(prediction_image):
    # Get the class with the highest probability
    predicted_classes = np.argmax(prediction_image, axis=-1)
    # Create an empty RGB image
    color_image = np.zeros(
        (predicted_classes.shape[0], predicted_classes.shape[1], 3), dtype=np.uint8
    )
    # Map each class to the corresponding color
    for class_index, color in color_map_rgb.items():
        color_image[predicted_classes == class_index] = color
    return color_image


# Function to make predictions and save visualizations
def predict_and_save_visualizations(model, source_folder, decoder_source_folder, save_folder, num_images=10):
    source_images = sorted(
        [
            os.path.join(source_folder, f)
            for f in os.listdir(source_folder)
            if f.endswith(".png")
        ]
    )

    decoder_source_images = sorted(
        [
            os.path.join(decoder_source_folder, f)
            for f in os.listdir(decoder_source_folder)
            if f.endswith(".png")
        ]
    )

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i in range(min(num_images, len(source_images))):
        source_image_path = source_images[i]
        decoder_source_image_path = decoder_source_images[i]
        source_image = load_image(source_image_path)
        decoder_source_image = load_image(decoder_source_image_path)
        source_image_expanded = tf.expand_dims(source_image, 0)  # Add batch dimension
        decoder_source_image_expanded = tf.expand_dims(decoder_source_image, 0)  # Add batch dimension

        prediction = model.predict([source_image_expanded, decoder_source_image_expanded])
        prediction_image = tf.squeeze(prediction, 0)  # Remove batch dimension
        prediction_image = prediction_image.numpy()  # Convert to numpy array

        # Map the prediction to colors
        prediction_image_colored = map_predictions_to_colors(prediction_image)

        # Convert source image to uint8 for consistency
        source_image_uint8 = tf.image.convert_image_dtype(
            source_image, dtype=tf.uint8
        ).numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(source_image_uint8.squeeze(), cmap="gray")
        axes[0].set_title("Source Image")
        axes[0].axis("off")

        axes[1].imshow(prediction_image_colored)
        axes[1].set_title("Prediction Image")
        axes[1].axis("off")

        save_path = os.path.join(save_folder, f"comparison_{i}.png")
        plt.savefig(save_path)
        plt.close(fig)


# Example usage
save_folder = "pred"

source_folder = "/root/arc/largest_val"
source_target_source = "/root/arc/largest_val_source"
# Assuming keras_model is your trained model
predict_and_save_visualizations(keras_model, source_folder, source_target_source, save_folder, num_images=22)
keras_model.save("saved_model.h5")
