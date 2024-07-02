import tensorflow as tf

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

        self.position_embedding_x_fn = tf.keras.layers.Embedding(input_dim=max_height, output_dim=d_model)
        self.position_embedding_y_fn = tf.keras.layers.Embedding(input_dim=max_width, output_dim=d_model)

        self.reshape1 = tf.keras.layers.Reshape((max_height * max_width, d_model))
        self.add = tf.keras.layers.Add()
        self.dense = tf.keras.layers.Dense(d_model)
        self.reshape2 = tf.keras.layers.Reshape((max_height, max_width, d_model))

    def call(self, inputs):
        position_embedding_x = self.position_embedding_x_fn(tf.range(self.max_height))
        position_embedding_y = self.position_embedding_y_fn(tf.range(self.max_width))
        
        position_embedding_x = tf.expand_dims(position_embedding_x, 1)  # Shape: (max_height, 1, d_model)
        position_embedding_y = tf.expand_dims(position_embedding_y, 0)  # Shape: (1, max_width, d_model)
        
        x = position_embedding_x + position_embedding_y  + self.dense(inputs)


        return x

# Define the input shape
input_shape = (30, 30, 1)
max_height = 30
max_width = 30
d_model = 128
num_heads = 8
key_dim = 16
attention_dropout = 0.1
num_attention_heads = 8
inputs = tf.keras.layers.Input(shape=input_shape)

# Instantiate the model
pos_embedding = PositionEmbeddingModel(max_height=max_height, max_width=max_width, d_model=d_model) 
self_attn = SelfAttention(num_heads= num_heads, key_dim= key_dim, attention_dropout= attention_dropout)
x = pos_embedding(inputs)

for _ in range(3):
    x = self_attn(x)


# Reshape and add dense layers for classification
x = tf.keras.layers.Reshape((max_height, max_width, d_model))(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(10, activation='softmax')(x)

# Create the Keras model
keras_model = tf.keras.Model(inputs=inputs, outputs=x)

# Compile the model
keras_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
keras_model.summary()





from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data
x_train = tf.image.resize(x_train[..., tf.newaxis], (30, 30)) / 255.0
x_test = tf.image.resize(x_test[..., tf.newaxis], (30, 30)) / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# Train the model
keras_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

