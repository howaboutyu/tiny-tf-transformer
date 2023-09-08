import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from utils import visualize_detections

PAD_VALUE = -1 

# TODO: DELETE 
#int2str = dataset_info.features["objects"]["label"].int2str
#print(int2str)
#
#names_list = tf.constant(dataset_info.features["objects"]["label"].names)
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

    # Resize the bounding boxes
    w_ratio = tf.cast(new_width, tf.float32) / tf.cast(width, tf.float32)
    h_ratio = tf.cast(new_height, tf.float32) / tf.cast(height, tf.float32)

    bboxes = tf.cast(bboxes, tf.float32)
    x_min = bboxes[:, 0] * w_ratio
    y_min = bboxes[:, 1] * h_ratio
    x_max = bboxes[:, 2] * w_ratio
    y_max = bboxes[:, 3] * h_ratio

    bboxes = tf.stack([x_min, y_min, x_max, y_max], axis=-1)

    return image, bboxes

    
def preprocess_fn( data, max_side=512, num_bins=256):

    label = data["objects"]["label"]
    image = data["image"]
    bboxes = data["objects"]["bbox"]

    # [x,y,width,height]
    bboxes = tf.cast( bboxes, tf.float32 )
    image = tf.cast( image, tf.float32 )

    # get shape
    image_shape = tf.shape( image )  
    image_width = tf.cast( image_shape[1], tf.float32 )
    image_height = tf.cast( image_shape[0], tf.float32 )

    bboxes = tf.cast( bboxes, tf.float32 )

    x_min = bboxes[:,1] * image_width
    y_min = bboxes[:,0] * image_height
    x_max = bboxes[:,3] * image_width
    y_max = bboxes[:,2] * image_height

    bboxes = tf.stack( [x_min,y_min,x_max,y_max], axis=1 )
    bboxes = tf.cast( bboxes, tf.int32 )

    image, bboxes = resize_image(image, bboxes, max_side=max_side) 

    # pad image
    image = tf.image.pad_to_bounding_box( image, 0, 0, max_side, max_side)

    bboxes = coord_to_bins( bboxes, max_side, num_bins=num_bins )
    bboxes = tf.cast(bboxes, tf.int32)

    return image, bboxes, label, image_shape

def coord_to_bins(coord, image_dim, num_bins):
    bin = tf.cast(  coord / (image_dim / (num_bins - 1) ), tf.int32 )
    return bin

def bins_to_coord(bin, image_dim, num_bins):
    if isinstance(bin, tf.Tensor):
        bin = bin.numpy()
    coord = (bin-1) * (image_dim / num_bins)
    return coord

def format_fn(image, bboxes, label, image_shape, EOS_TOKEN, max_objects=40):

    # normalize image 
    image = image / 255.0

    num_objects = tf.shape( bboxes )[0]
    sequence_ids = tf.range( num_objects )

    # shuffle
    random_index = tf.random.shuffle( sequence_ids )

    random_bboxes = tf.gather( bboxes, random_index )
    random_labels = tf.gather( label, random_index ) 

    random_bboxes = tf.reshape( random_bboxes, (-1,4) )
    random_labels = tf.reshape( random_labels, (-1,1) )


    random_labels = tf.cast(random_labels, tf.int32)
    target = tf.concat( [random_bboxes, random_labels], axis=1 )

    # flatten target
    target = tf.reshape( target, (-1,) )
    target = tf.concat( [target, [EOS_TOKEN]], axis=0 )

    decoder_input = target[..., :-1]
    decoder_output = target[..., 1:]

    # limit input and target  
    if tf.shape(decoder_output)[0] > max_objects*5:
        decoder_input = decoder_input[..., :max_objects*5]
        decoder_output = decoder_output[..., :max_objects*5]


    elif tf.shape(decoder_output)[0] < max_objects*5:
        pad = max_objects*5 - tf.shape(decoder_output)[0]
        decoder_input = tf.pad( decoder_input, [[0,pad]], constant_values=PAD_VALUE )
        decoder_output = tf.pad( decoder_output, [[0,pad]], constant_values=PAD_VALUE )


    return (image, decoder_input), decoder_output

