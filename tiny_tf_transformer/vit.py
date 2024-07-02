import tensorflow as tf
import os
from tensorflow.keras import mixed_precision

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


class PositionEmbeddingModel(tf.keras.Model):
    def __init__(self, max_height=30, max_width=30, d_model=128):
        super(PositionEmbeddingModel, self).__init__()
        self.max_height = max_height
        self.max_width = max_width
        self.d_model = d_model

        self.position_embedding_x_fn = tf.keras.layers.Embedding(
            input_dim=max_height, output_dim=d_model
        )
        self.position_embedding_y_fn = tf.keras.layers.Embedding(
            input_dim=max_width, output_dim=d_model
        )
        self.class_embedding_fn = tf.keras.layers.Embedding(
            input_dim=10, output_dim=d_model
        )

        self.dense = tf.keras.layers.Dense(d_model)
        self.conv = tf.keras.layers.Conv2D(
            d_model,
            (2, 2),
            strides=(2, 2),
            padding="same",
        )

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.int8)

        print(inputs.shape)
        x = tf.one_hot(inputs, 10, axis=-1)
        # x = tf.squeeze(x, -2)
        x = self.conv(x)

        w = x.shape[1]
        h = x.shape[2]

        position_embedding_x = self.position_embedding_x_fn(
            tf.range(self.max_height // 2)
        )
        position_embedding_y = self.position_embedding_y_fn(
            tf.range(self.max_width // 2)
        )

        position_embedding_x = tf.expand_dims(
            position_embedding_x, 1
        )  # Shape: (max_height, 1, d_model)
        position_embedding_y = tf.expand_dims(
            position_embedding_y, 0
        )  # Shape: (1, max_width, d_model)
        pos_x = position_embedding_x + position_embedding_y
        x = pos_x + x 

        print("xshape", x.shape)

        return x


# Define the input shape
input_shape = (20 * 3, 20 * 2)
max_height = 20 * 3
max_width = 20 * 2
d_model = 128
d_ff = 64
num_heads = 8
key_dim = 16
attention_dropout = 0.1
num_attention_heads = 8
inputs = tf.keras.layers.Input(shape=input_shape)

# Instantiate the model
pos_embedding = PositionEmbeddingModel(
    max_height=max_height, max_width=max_width, d_model=d_model
)
x_original = pos_embedding(inputs)

x = x_original


for _ in range(4):
    encoding_layer = EncoderLayer(num_heads=num_heads, d_model=d_model, d_ff=d_ff)
    # cross_attention = CrossAttention(num_heads=num_heads, key_dim=d_model)
    x = encoding_layer(x)
    # x = cross_attention(x, x_original)

# Reshape and add dense layers for classification
# x = tf.keras.layers.Reshape((max_height, max_width, d_model))(x)
# x = tf.keras.layers.GlobalAveragePooling2D()(x)
# x = tf.keras.layers.Dense(128, activation='relu')(x)
x = x[:, -20:, :20]
x = tf.keras.layers.Dense(10, activation="linear")(x)

# Create the Keras model
keras_model = tf.keras.Model(inputs=inputs, outputs=x)

keras_model.load_weights('saved_model.h5')
# Compile the model
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4, decay_steps=200, decay_rate=0.9
)

opt = tf.keras.optimizers.AdamW(learning_rate=1e-4)
keras_model.compile(optimizer=opt, loss=loss, run_eagerly=False)

# Print the model summary
keras_model.summary()


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


def load_data(source_folder, target_folder):
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

    print(f"number images: {len(source_images)}")

    target_map = lambda x: load_image(x, True)
    source_dataset = tf.data.Dataset.from_tensor_slices(source_images).map(
        load_image, num_parallel_calls=tf.data.AUTOTUNE
    )
    target_dataset = tf.data.Dataset.from_tensor_slices(target_images).map(
        target_map, num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = tf.data.Dataset.zip((source_dataset, target_dataset))
    dataset = (
        dataset.shuffle(buffer_size=1024)
        .batch(128)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return dataset


source_folder = "/root/arc/largest"
target_folder = "/root/arc/largest_target"

val_source_folder = "/root/arc/largest_val"
val_target_folder = "/root/arc/largest_val_target"


# Load the dataset
train_dataset = load_data(source_folder, target_folder)
val_dataset = load_data(val_source_folder, val_target_folder)


# Train the model
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

keras_model.fit(
    train_dataset,
    epochs=50,
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
def predict_and_save_visualizations(model, source_folder, save_folder, num_images=10):
    source_images = sorted(
        [
            os.path.join(source_folder, f)
            for f in os.listdir(source_folder)
            if f.endswith(".png")
        ]
    )

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i in range(min(num_images, len(source_images))):
        source_image_path = source_images[i]
        source_image = load_image(source_image_path)
        source_image_expanded = tf.expand_dims(source_image, 0)  # Add batch dimension

        prediction = model.predict(source_image_expanded)
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
# Assuming keras_model is your trained model
predict_and_save_visualizations(keras_model, source_folder, save_folder, num_images=10)
keras_model.save("saved_model.h5")
