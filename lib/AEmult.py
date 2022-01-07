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

from tensorflow_probability import distributions as tfd
def zinb(r, logits, dropout):
    catDist = tfd.Categorical(logits=tf.stack([dropout, 1.0 - dropout], axis=-1))
    det = tfd.Deterministic(loc=tf.zeros_like(r))
    negbin = tfd.NegativeBinomial(r, logits=logits)
    return tfd.Mixture(cat=catDist,
                       components=[det, negbin])

def binomMix(n_i, logits_cat, logits_binom, k):
    catDist = tfd.Categorical(logits = tf.stack(tf.split(logits_cat, k), axis=-1))
    binomDist = tfd.Binomial(n_i, tf.stack(tf.split(logits_binom, k), axis=-1))
    return tfd.MixtureSameFamily(catDist, binomDist)






class AE:
    # Variationnal autoencoder
    def __init__(self, mode="Binary", latent_dim=32, loss=None, dropoutRate=0.15):
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
        f = tf.keras.layers.BatchNormalization(renorm=True)(f)
        return tf.keras.layers.Dropout(0.0)(f)

    def reconstructionEntropy(self, y_true, y_pred):
        r, logits = tf.split(y_pred, [1, self.dim], 1)
        dist = tfd.DirichletMultinomial(r, concentration=logits)
        loss = -dist.log_prob(y_true)
        loss = tf.reduce_sum(loss, axis=1)
        return tf.reduce_mean(loss)

    def createModel(self, X):
        # Creates model : vae and encoder only
        exampleCount = X.shape[0]
        original_dim = X.shape[1]
        self.dim = original_dim
        self.beta = tf.Variable(0.0, trainable=False)
        # Encoding layers
        x = tf.keras.layers.Input(shape=(original_dim,))
        # Latent space
        z = tf.keras.layers.Dense(self.latent_dim, kernel_initializer="zeros")(x)
        n_i = tf.math.reduce_sum(x, axis=1, keepdims=True)
        conc = tf.keras.layers.Dense(original_dim, tf.math.exp, kernel_initializer="zeros")(z)
        output = tf.concat([n_i, conc], axis=1)


        self.vae = tf.keras.Model(inputs=x, outputs=output)
        self.encoder = tf.keras.Model(inputs=x, outputs=z)
        # Use Adam with warm restarts and varying learning rate. Restart every 5 epochs
        cosineRestartLen = int(exampleCount/self.batchSize*5)
        lrSchedule = CosineSchedule(cosineRestartLen, 1e-3)
        optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-6)
        if self.variableType == "Binary":
            self.vae.compile(optimizer=optimizer, loss=self.reconstructionEntropy, metrics=[self.reconstructionEntropy])
        else:
            self.vae.compile(optimizer=optimizer, loss="mse", metrics=["mse"])

    def fit_transform(self, Xin, Xout):
        # Create and fit model according on data
        # Returns data encoded in latent space
        Xin = Xin.astype("float32")
        Xout = Xout.astype("float32")
        # Good rule of thumb batch size
        # Large batch sizes on small datasets result in poor performance
        # Small batch size on large datasets take really long to train
        self.batchSize = int(np.sqrt(len(Xin)))
        self.numBatches = int(np.sqrt(len(Xin)/self.batchSize)*500)
        self.numEpochs = 1000
        # Generate model
        self.createModel(Xin)
        # Stop training when the model stops improving
        esDuration = 11
        es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=esDuration, restore_best_weights=True)
        # Increase normality constraint on latent space
        csv = tf.keras.callbacks.CSVLogger('training.log')
        # self.vae.summary()
        # Fit model and continue script on keyboard interrupt
        try:
            # Split between training and validation set to avoid overfitting
            self.vae.fit(Xin, Xout, batch_size=self.batchSize, 
                         epochs=self.numEpochs, verbose=2, callbacks=[es])        

        except KeyboardInterrupt:
            print("Force stopped training")
        # Returns data projected in the latent space
        return self.encoder.predict(Xin, batch_size=512)


    def transform(self, X):
        return self.encoder.predict(X, batch_size=512)

    def reconstruct(self, X):
        return np.clip(expit(self.vae.predict(X, batch_size=512)),0.0,1.0)
