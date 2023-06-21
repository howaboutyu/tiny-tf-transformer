import tensorflow as tf


def masked_sparse_categorical_cross_entropy(y_true, y_pred):
    """
    Calculates the masked sparse categorical cross-entropy loss.

    Args:
        y_true: The true labels, where each label is an integer.
        y_pred: The predicted probabilities for each class.

    Returns:
        The masked sparse categorical cross-entropy loss.
    """

    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss_ = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=True
    )
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_) / tf.reduce_sum(mask)


def masked_sparse_categorical_accuracy(y_true, y_pred):
    """
    Calculates the masked sparse categorical accuracy.

    Args:
        y_true: The true labels, where each label is an integer.
        y_pred: The predicted probabilities for each class.

    Returns:
        The masked sparse categorical accuracy.
    """

    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    y_true = tf.cast(y_true, dtype=tf.int32)
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, dtype=tf.int32)
    correct = tf.cast(tf.equal(y_true, y_pred), dtype=tf.float32)
    mask = tf.cast(mask, dtype=correct.dtype)
    correct *= mask
    return tf.reduce_sum(correct) / tf.reduce_sum(mask)
