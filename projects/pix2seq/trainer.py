from dataset import preprocess_fn, format_fn 
import tensorflow_datasets as tfds
import tensorflow as tf

(train_ds, val_ds), dataset_info = tfds.load( "coco/2017", split=["train", "validation"], with_info=True, data_dir="~/data" )


max_side = 512
num_bins = 256 
batch_size = 20
max_objects = 40

EOS_TOKEN = num_bins + 1

train_ds = train_ds.map( 
    lambda x: preprocess_fn(x, max_side=max_side, num_bins=num_bins), 
    num_parallel_calls=tf.data.experimental.AUTOTUNE )

train_ds = train_ds.map(
    lambda image, bboxes, label, image_shape: format_fn(image, bboxes, label, image_shape, EOS_TOKEN, max_objects=max_objects),
    )

train_ds = train_ds.batch( batch_size )

#for t in train_ds.take(10):
#    (image, decoder_input), decoder_output = t
#    print(np.max(decoder_output))
