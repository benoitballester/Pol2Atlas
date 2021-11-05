import tensorflow as tf
from tensorflow_addons.optimizers import AdamW
import numpy as np


class CosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, restart, baseLr):
    super(CosineSchedule, self).__init__()
    self.restart = restart
    self.lr = baseLr

  def __call__(self, step):
    arg = tf.math.mod(step/self.restart, 1.0)*np.pi
    # Slow increase in lr at the start (2 first restarts)
    arg = tf.where(step < self.restart*2, np.pi-tf.math.mod(step/self.restart/2.0, 1.0)*np.pi, arg)
    return (tf.math.cos(arg)*0.4999+0.5001)*self.lr

def layer(n, x, dropout=0.5):
    f = tf.keras.layers.Dense(n, "gelu")(x)
    f = tf.keras.layers.BatchNormalization()(f)
    f = tf.keras.layers.Dropout(dropout)(f)
    return f

def mlp(x_shape, n_labels, layers=[512,512,512], dropout_in=0.15, dropout=0.5):
    x_in = tf.keras.layers.Input(x_shape[1])
    f = tf.keras.layers.Dropout(dropout_in)(x_in)
    for l in layers:
        f = layer(l, f, dropout)
    pred = tf.keras.layers.Dense(n_labels, "softmax")(f)
    model = tf.keras.Model(x_in, pred)
    lrSchedule = CosineSchedule(100, 1e-3)
    optimizer = AdamW(learning_rate=lrSchedule, weight_decay=1e-4)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy")
    return model

    
