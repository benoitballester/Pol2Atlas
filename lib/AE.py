import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LambdaCallback
import pickle
from numba import jit, prange
from tensorflow_addons.optimizers import LAMB, AdamW
from scipy.special import expit
import tensorflow_probability as tfp

class CosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, restart, baseLr, batch_size):
    super(CosineSchedule, self).__init__()
    self.restart = restart
    self.lr = baseLr

  def __call__(self, step):
    arg = tf.math.mod(step/self.restart, 1.0)*np.pi
    # Slow increase in lr at the start (2 first restarts)
    arg = tf.where(step < self.restart*2, np.pi-tf.math.mod(step/self.restart/2.0, 1.0)*np.pi, arg)
    return (tf.math.cos(arg)*0.499+0.501)*self.lr


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



class AE:
    """
    AutoEncoder


    Parameters
    ----------
    

    """
    def __init__(self, latent_dim=10, dropoutRate=0.3, layerSize=2048, numLayers=3):
        self.latent_dim = latent_dim
        self.drop = dropoutRate
        self.layerSize = layerSize
        self.numLayers = numLayers

    def layer(self, n, i):
        f = tf.keras.layers.Dense(n, activation="selu", kernel_initializer="lecun_normal")(i)
        return f

    def createModel(self, X):
        # Creates model : vae and encoder only
        exampleCount = X.shape[0]
        original_dim = X.shape[1]
        probs = np.mean(X, axis=0)
        mu = tf.reduce_mean(X, axis=0)
        sd = tf.math.reduce_std(X)+1e-5
        # Encoding layers
        x = tf.keras.layers.Input(shape=(original_dim,))
        h = (x - mu)/sd
        h = tf.keras.layers.Dropout(self.drop)(h)
        # Encoding layers
        for i in [self.layerSize]*self.numLayers:
            h = self.layer(i, h)
        # Latent space
        z_mu = tf.keras.layers.Dense(self.latent_dim)(h)
        z_mu = tf.keras.layers.BatchNormalization()(z_mu)
        # Decoding layers
        h = z_mu
        for i in [self.layerSize]*self.numLayers:
            h = self.layer(i, h)
        x_pred = tf.keras.layers.Dense(original_dim)(h)
        dist = tfp.distributions.Bernoulli(logits=x_pred)
        loss = tf.reduce_sum(-dist.log_prob(x) / probs, axis=1, keepdims=True)
        self.vae = tf.keras.Model(inputs=x, outputs=loss)
        self.encoder = tf.keras.Model(inputs=x, outputs=z_mu)
        # Use Adam with warm restarts and varying learning rate. Restart every 5 epochs
        cosineRestartLen = int(exampleCount/self.batchSize * 5)
        lrSchedule = CosineSchedule(cosineRestartLen, 1e-3, self.batchSize)
        optimizer = AdamW(learning_rate=lrSchedule, weight_decay=1e-4)
        self.vae.compile(optimizer=optimizer, loss=lambda _, t: tf.reduce_mean(t))

    def fit_transform(self, X):
        # Create and fit model according on data
        # Returns data encoded in latent space
        X = X.astype("float32")
        # Good rule of thumb batch size
        # Large batch sizes on small datasets result in poor performance
        # Small batch size on large datasets take really long to train
        self.batchSize = int(np.sqrt(len(X)))
        # Generate model
        self.createModel(X)
        # Stop training when the model stops improving
        esDuration = 21
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=esDuration, restore_best_weights=True)
        # Increase normality constraint on latent space
        csv = tf.keras.callbacks.CSVLogger('training.log')
        # self.vae.summary()
        # Fit model and continue script on keyboard interrupt
        try:
            # Split between training and validation set to avoid overfitting
            Xtrain, Xval = train_test_split(X, test_size=0.05, shuffle=True, random_state=42)
            wu_cb = LambdaCallback(on_epoch_end=lambda epoch, logs:warmup(epoch, self.beta))
            # self.vae.fit(Xtrain, Xtrain, batch_size=self.batchSize, epochs=50, callbacks=[wu_cb], verbose=2)   
            self.vae.fit(Xtrain, Xtrain, batch_size=self.batchSize, epochs=1500, validation_data=(Xval, Xval), callbacks=[es, csv], verbose=2)        
        except KeyboardInterrupt:
            print("Force stopped training")
        # Returns data projected in the latent space
        return self.encoder.predict(X, batch_size=512)


    def transform(self, X):
        return self.encoder.predict(X, batch_size=512)

    def reconstruct(self, X):
        return np.clip(expit(self.vae.predict(X, batch_size=512)),0.0,1.0)
