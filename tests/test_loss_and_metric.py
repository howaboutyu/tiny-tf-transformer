import context

import pytest
import numpy as np
import tensorflow as tf

from tiny_tf_transformer.losses_and_metrics import (
    masked_sparse_categorical_accuracy,
    masked_sparse_categorical_cross_entropy,
)


def test_masked_sprase_categorical_accuracy():
    # no padding

    y_true = [[1, 2]]
    y_pred = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]]]

    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)

    acc_no_pad = masked_sparse_categorical_accuracy(y_true, y_pred)

    y_true = [[1, 2, 0]]
    y_pred = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1]]]

    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)

    acc_with_pad = masked_sparse_categorical_accuracy(y_true, y_pred)

    assert acc_no_pad == acc_with_pad
