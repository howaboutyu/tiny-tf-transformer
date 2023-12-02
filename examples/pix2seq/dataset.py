import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from utils import visualize_detections

PAD_VALUE = 0

# int2str = dataset_info.features["objects"]["label"].int2str
# print(int2str)
#
# names_list = tf.constant(dataset_info.features["objects"]["label"].names)
# mapping function from tf int to string


def resize_image(image, bboxes, max_side=512):
    """
    Resize image with unchanged aspect ratio.
    Inputs:
        image: 3-D Tensor of shape (H, W, C)
        bboxes: 2-D Tensor of shape (num_bboxes, 4)
        max_side: an int, the max side of the image.
    """

    # Get current shape
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    # Get the largest side of image w.r.t. bboxes
    def _largest_size():
        largest_side = tf.maximum(height, width)
        scale = tf.cast(max_side, tf.float32) / tf.cast(largest_side, tf.float32)
        new_height = tf.cast(tf.cast(height, tf.float32) * scale, tf.int32)
        new_width = tf.cast(tf.cast(width, tf.float32) * scale, tf.int32)
        return new_height, new_width

    new_height, new_width = tf.cond(
        tf.greater(height, width), lambda: _largest_size(), lambda: _largest_size()
    )

    # Resize the image with the computed size
    image = tf.image.resize(image, (new_height, new_width))

    w_ratio = tf.cast(new_width, tf.float32) / tf.cast(max_side, tf.float32)
    h_ratio = tf.cast(new_height, tf.float32) / tf.cast(max_side, tf.float32)

    bboxes = tf.cast(bboxes, tf.float32)
    x_min = bboxes[:, 0] * w_ratio
    y_min = bboxes[:, 1] * h_ratio
    x_max = bboxes[:, 2] * w_ratio
    y_max = bboxes[:, 3] * h_ratio

    bboxes = tf.stack([x_min, y_min, x_max, y_max], axis=-1)

    return image, bboxes


def preprocess_fn(data, max_side=512, num_bins=256):
    label = data["objects"]["label"]
    image = data["image"]
    bboxes = data["objects"]["bbox"]

    # [x,y,width,height]
    bboxes = tf.cast(bboxes, tf.float32)
    image = tf.cast(image, tf.float32)

    # get shape
    image_shape = tf.shape(image)
    image_width = tf.cast(image_shape[1], tf.float32)
    image_height = tf.cast(image_shape[0], tf.float32)

    bboxes = tf.cast(bboxes, tf.float32)

    # normalize bboxes
    x_min = bboxes[:, 1]
    y_min = bboxes[:, 0]
    x_max = bboxes[:, 3]
    y_max = bboxes[:, 2]

    bboxes = tf.stack([x_min, y_min, x_max, y_max], axis=1)

    #image = tf.image.resize(image, (max_side, max_side))

    image, bboxes = resize_image(image, bboxes, max_side=max_side)

    # pad image
    image = tf.image.pad_to_bounding_box(image, 0, 0, max_side, max_side)

    # quantize bboxes
    bboxes = quantize(bboxes, bins=num_bins)

    return image, bboxes, label, image_shape


def quantize(x, bins=1000):
    # x is a real number between [0, 1]
    # returns an integer between [0, bins-1]
    if isinstance(x, tf.Tensor):
        return tf.cast(x * (bins - 1), tf.int32)
    else:
        return int(x * (bins - 1))


def dequantize(x, bins=1000):
    # x is an integer between [0, bins-1]
    # returns a real number between [0, 1]
    if isinstance(x, tf.Tensor):
        return tf.cast(x, tf.float32) / (bins - 1)
    else:
        return float(x) / (bins - 1)


def format_fn(image, bboxes, label, image_shape, sos_token, eos_token, max_objects=40):
    num_objects = tf.shape(bboxes)[0]
    sequence_ids = tf.range(num_objects)

    # shuffle
    random_index = tf.random.shuffle(sequence_ids)

    random_bboxes = tf.gather(bboxes, random_index)
    random_labels = tf.gather(label, random_index)

    random_bboxes = tf.reshape(random_bboxes, (-1, 4))
    random_labels = tf.reshape(random_labels, (-1, 1))

    random_labels = tf.cast(random_labels, tf.int32)
    target = tf.concat([random_bboxes, random_labels], axis=1)

    """
    bboxes = tf.reshape(bboxes, (-1, 4))
    label = tf.reshape(label, (-1, 1))
    label = tf.cast(label, tf.int32)
    target = tf.concat([bboxes, label], axis=1)
    """
    # add one to labels as 0 is reserved for padding
    target += 1

    # flatten target
    target = tf.reshape(target, (-1,))
    target = tf.concat([[sos_token], target, [eos_token] * 5], axis=0)

    decoder_input = target[..., :-1]
    decoder_output = target[..., 1:]

    # limit input and target
    if tf.shape(decoder_output)[0] > max_objects * 5:
        decoder_input = decoder_input[..., : max_objects * 5]
        decoder_output = decoder_output[..., : max_objects * 5]

    elif tf.shape(decoder_output)[0] < max_objects * 5:
        pad = max_objects * 5 - tf.shape(decoder_output)[0]
        decoder_input = tf.pad(decoder_input, [[0, pad]], constant_values=PAD_VALUE)
        decoder_output = tf.pad(decoder_output, [[0, pad]], constant_values=PAD_VALUE)

    return (image, decoder_input), decoder_output


def sequence_decoder(decoder_output, eos_token, num_bins):
    """
    Decodes the output of a sequence decoder into bounding boxes and labels.

    Args:
    decoder_output (Tensor): The output from the decoder model.
    eos_token (int): The token representing the end of a sequence.
    num_bins (int): Number of bins used for dequantization.

    Returns:
    tuple: A tuple containing lists of filtered bounding boxes and labels.
    """

    # Extract the batch size from the decoder's output shape
    batch_size = tf.shape(decoder_output)[0]

    # Reshape the decoder output to separate the bounding box coordinates and labels
    decoder_output = tf.reshape(decoder_output, (batch_size, -1, 5))

    # Extract bounding boxes and labels from the decoder output
    bboxes = decoder_output[..., :4]  # shape: (batch_size, num_objects, 4)
    labels = decoder_output[..., 4:]  # shape: (batch_size, num_objects, 1)

    # Initialize lists to store the filtered bounding boxes and labels
    bboxes_filtered = []
    labels_filtered = []

    # Process each item in the batch
    for i in range(batch_size):
        # Identify the first occurrence of the end-of-sequence token in labels
        index = tf.where(tf.equal(labels[i], eos_token))

        # Check if the end-of-sequence token is found
        if tf.shape(index)[0] != 0:
            # Use the index of the EOS token to filter the bounding boxes and labels
            index = index[0, 0]
            bboxes_tmp = bboxes[i, :index] - 1
            labels_tmp = labels[i, :index] - 1
        else:
            # If EOS token is not found, use all boxes and labels
            bboxes_tmp = bboxes[i] - 1
            labels_tmp = labels[i] - 1

        # Dequantize the bounding boxes
        bboxes_tmp = dequantize(bboxes_tmp, bins=num_bins)

        # Append the processed boxes and labels to the respective lists
        bboxes_filtered.append(bboxes_tmp)
        labels_filtered.append(labels_tmp)

    # Return the filtered bounding boxes and labels
    return bboxes_filtered, labels_filtered
