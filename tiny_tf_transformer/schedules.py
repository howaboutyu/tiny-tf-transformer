import tensorflow as tf


class TransformerLearningRateSchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule
):
    """Original schedule in original paper: https://arxiv.org/abs/1706.03762"""

    def __init__(self, d_model, warmup_steps=4000):
        super(TransformerLearningRateSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
