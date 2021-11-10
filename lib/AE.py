import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LambdaCallback
import pickle
from numba import jit, prange
from tensorflow_addons.optimizers import LAMB, AdamW, SGDW
from scipy.special import expit
import tensorflow_probability as tfp
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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

class CosineScheduleSingle(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, numSteps, baseLr):
    super(CosineScheduleSingle, self).__init__()
    self.restart = numSteps
    self.lr = baseLr

  def __call__(self, step):
    arg = step/self.restart*np.pi*1.9999 + np.pi
    return (tf.math.cos(arg)*0.499999+0.500001)*self.lr

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """
    dist = tfp.distributions.Bernoulli(logits=y_pred)
    loss = tf.reduce_mean(-dist.log_prob(y_true), axis=1)
    return tf.reduce_mean(loss)

class KLDivergenceLayer(tf.keras.layers.Layer):
    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, klWeight, *args, **kwargs):
        self.is_placeholder = True
        self.klWeight = klWeight
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)
        klLoss = K.mean(kl_batch)
        self.add_loss(klLoss*self.klWeight, inputs=inputs)
        return inputs


def dice_loss(y_true, y_pred):
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.math.sigmoid(y_pred)
  numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=1)
  denominator = tf.reduce_sum(y_true + y_pred, axis=1)

  return tf.reduce_mean(1 - numerator / denominator)

def warmup(epoch, beta):
    beta.assign(min(epoch/20.0+0.01,1.0))


class AE:
    # Variationnal autoencoder
    def __init__(self, mode="Binary", latent_dim=32, loss=None, dropoutRate=0.0):
        # Layers = do not work atm
        # If not specified will be set automatically
        # Mode : "Binary" / "Real". Binary for values between 0-1 and "Real" for other values
        # latent_dim : number of dimensions of the latent space
        self.latent_dim = latent_dim
        self.variableType = mode
        self.loss = loss
        self.drop = dropoutRate

    def layer(self, n, i):
        f = tf.keras.layers.Dense(n, activation="gelu")(i)
        return tf.keras.layers.BatchNormalization()(f)

    def createModel(self, X):
        # Creates model : vae and encoder only
        exampleCount = X.shape[0]
        original_dim = X.shape[1]
        self.beta = tf.Variable(0.0, trainable=False)
        # Encoding layers
        x = tf.keras.layers.Input(shape=(original_dim,))
        h = tf.keras.layers.Dropout(self.drop)(x)
        # Encoding layers
        h = tf.keras.layers.BatchNormalization()(h)
        for i in [2048]*3:
            h = self.layer(i, h)
        # Latent space
        z = tf.keras.layers.Dense(self.latent_dim)(h)
        h = tf.keras.layers.BatchNormalization()(z)
        # Decoding layers
        for i in [2048]*3:
            h = self.layer(i, h)
        x_pred = tf.keras.layers.Dense(original_dim)(h)

        self.vae = tf.keras.Model(inputs=x, outputs=x_pred)
        self.encoder = tf.keras.Model(inputs=x, outputs=z)
        # Use Adam with warm restarts and varying learning rate. Restart every 5 epochs
        cosineRestartLen = int(exampleCount/self.batchSize * 5)
        lrSchedule = CosineSchedule(cosineRestartLen, 1e-3)
        optimizer = AdamW(learning_rate=lrSchedule, weight_decay=1e-6)
        if self.variableType == "Binary":
            self.vae.compile(optimizer=optimizer, loss=nll, metrics=[nll,dice_loss,  "cosine_similarity"])
        else:
            self.vae.compile(optimizer=optimizer, loss="mse", metrics=["mse"])

    def fit_transform(self, X):
        # Create and fit model according on data
        # Returns data encoded in latent space
        X = X.astype("float32")
        # Good rule of thumb batch size
        # Large batch sizes on small datasets result in poor performance
        # Small batch size on large datasets take really long to train
        self.batchSize = int(np.sqrt(len(X)))
        self.numBatches = int(np.sqrt(len(X)/self.batchSize)*500)
        self.numEpochs = 1000
        # Generate model
        self.createModel(X)
        # Stop training when the model stops improving
        esDuration = 31
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=esDuration, restore_best_weights=True)
        # Increase normality constraint on latent space
        csv = tf.keras.callbacks.CSVLogger('training.log')
        # self.vae.summary()
        # Fit model and continue script on keyboard interrupt
        try:
            # Split between training and validation set to avoid overfitting
            Xtrain, Xval = train_test_split(X, test_size=0.05, shuffle=True, random_state=42)
            self.vae.fit(Xtrain, Xtrain, batch_size=self.batchSize, validation_data=(Xval, Xval), 
                         epochs=self.numEpochs, verbose=2, callbacks=[es])        

        except KeyboardInterrupt:
            print("Force stopped training")
        # Returns data projected in the latent space
        return self.encoder.predict(X, batch_size=512)


    def transform(self, X):
        return self.encoder.predict(X, batch_size=512)

    def reconstruct(self, X):
        return np.clip(expit(self.vae.predict(X, batch_size=512)),0.0,1.0)
