import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split


class nb_loglike(tf.keras.layers.Layer):
    def __init__(self, units):
        super(nb_loglike, self).__init__()
        self.units = units
        self.inputTransform = tf.math.sqrt
        self.linkFunction = tf.math.exp

    def build(self, _):
        # Roughly start with a Poisson-like overdispersion
        self.logalpha = self.add_weight(name='alphas',shape=(self.units,),
                                        initializer=tf.keras.initializers.Constant(-0.4), 
                                        trainable=True)
                            
    def call(self, inputs):
        y_true, y_pred_lin = inputs
        y_pred = self.linkFunction(y_pred_lin)
        alphaInv = tf.math.exp(-self.logalpha*10)
        alpha = tf.math.exp(self.logalpha*10)
        # Tensorflow built-in nb logprob is not stable at all numerically
        l = tf.math.lgamma(y_true + alphaInv)
        l -= tf.math.lgamma(y_true+1)
        l -= tf.math.lgamma(alphaInv)
        l -= alphaInv * tf.math.log1p(alpha * y_pred)
        l += y_true * (tf.math.log(alpha * y_pred) - tf.math.log1p(alpha*y_pred))
        return -tf.reduce_mean(tf.reduce_sum(l, axis=1))

class bernoulli_loglike(tf.keras.layers.Layer):
    def __init__(self, units):
        super(bernoulli_loglike, self).__init__()
        self.units = units
        self.inputTransform = lambda x:x
        self.linkFunction = tf.nn.sigmoid
        
    def build(self, _):
        pass
                            
    def call(self, inputs):
        y_true, y_pred_lin = inputs
        y_pred = self.linkFunction(y_pred_lin)
        # Tensorflow built-in nb logprob is not stable at all numerically
        l = tf.math.lgamma(y_true)
        return -tf.reduce_mean(tf.reduce_sum(l, axis=1))


class glmPCA:
    def __init__(self, probModel, latentDim=50):
        if probModel == "NB":
            self.prob = nb_loglike
        if probModel == "Bernoulli":
            self.prob = bernoulli_loglike

        self.latentSize = latentDim

    def _buildModels(self, data):
        self.probModel = self.prob(data.shape[1])
        sf = tf.keras.layers.Input((1,))
        countsIn = tf.keras.layers.Input((data.shape[1],))
        logCounts = tf.math.log(0.01+countsIn/sf)
        latent = tf.keras.layers.Dense(1, use_bias=False)(logCounts)
        reconstructed = tf.keras.layers.Dense(data.shape[1])(latent) + sf
        l = self.probModel([countsIn, reconstructed])
        self.model = tf.keras.Model([countsIn,sf], l)
        self.model.compile(tf.keras.optimizers.Adam(), loss=lambda _, x:x)
        self.latentModel = tf.keras.Model([countsIn,sf], latent)
        self.denoiser = tf.keras.Model([countsIn,sf], self.probModel.linkFunction(reconstructed))
    
    def fit(self, data, sf, maxEpochs=500, batchSize="auto", es=10):
        self._buildModels(data)
        sfMat = sf[: ,None]
        es = tf.keras.callbacks.EarlyStopping("loss", patience=es, restore_best_weights=True)
        if batchSize == "auto":
            batchSize = int(np.sqrt(len(data)))
        self.model.fit(x=(data,sf), y=np.zeros((len(data),1)),
                        epochs=maxEpochs, batch_size=64, callbacks=[es], verbose=2)

    def fit_transform(self, data, sf, maxEpochs=500, batchSize="auto", es=10):
        self.fit(data, sf, maxEpochs, batchSize, es)
        return self.latentModel.predict((data,sf), batch_size=512)


class DenoiserAE:
    def __init__(self, probModel, layerSize="auto", latentSize="auto"):
        if probModel == "NB":
            self.prob = nb_loglike
        if probModel == "Bernoulli":
            self.prob = bernoulli_loglike
        self.layerSize = layerSize
        self.latentSize = latentSize

    def _buildModels(self, data):
        self.probModel = self.prob(data.shape[1])
        layerSize = int(np.power(np.prod(data.shape),0.25)*4.0)
        latentSize = int(layerSize/4+1)
        countsIn = tf.keras.layers.Input((data.shape[1],))
        m = tf.reduce_mean(countsIn, axis=1, keepdims=True)
        logCounts = self.probModel.inputTransform(countsIn/m)
        logCounts = tf.keras.layers.Dropout(0.5)(logCounts)
        e1 = tf.keras.layers.Dense(layerSize, "selu", kernel_initializer="lecun_normal")(logCounts)
        e2 = tf.keras.layers.Dense(layerSize, "selu", kernel_initializer="lecun_normal")(e1)
        e3 = tf.keras.layers.Dense(layerSize, "selu", kernel_initializer="lecun_normal")(e2)
        latent = tf.keras.layers.Dense(latentSize, kernel_initializer="lecun_normal")(e3)
        latent = tf.keras.layers.BatchNormalization(renorm=True)(latent)
        d1 = tf.keras.layers.Dense(layerSize, "selu", kernel_initializer="lecun_normal")(latent)
        d2 = tf.keras.layers.Dense(layerSize, "selu", kernel_initializer="lecun_normal")(d1)
        d3 = tf.keras.layers.Dense(layerSize, "selu", kernel_initializer="lecun_normal")(d2)
        reconstructed = tf.keras.layers.Dense(data.shape[1], kernel_initializer="lecun_normal")(d3) + tf.math.log(m)
        l = self.probModel([countsIn, reconstructed])
        self.model = tf.keras.Model(countsIn, l)
        self.model.compile(tfa.optimizers.AdamW(1e-6, 1e-3, amsgrad=True), loss=lambda _, x:x)
        self.latentModel = tf.keras.Model(countsIn, latent)
        self.denoiser = tf.keras.Model(countsIn, self.probModel.linkFunction(reconstructed))
    
    def fit(self, data, maxEpochs=500, batchSize="auto", es=10):
        self._buildModels(data)
        es = tf.keras.callbacks.EarlyStopping("val_loss", patience=es, restore_best_weights=True)
        x_train, x_test = train_test_split(data, test_size=0.05, random_state=42)
        if batchSize == "auto":
            batchSize = int(np.sqrt(len(data)))
        self.model.fit(x=x_train, y=np.zeros((len(x_train),1)), validation_data=(x_test, np.zeros((len(x_test),1))),
                        epochs=maxEpochs, batch_size=batchSize, callbacks=[es], verbose=2)

    def fit_transform(self, data, maxEpochs=500, batchSize="auto", es=5):
        self.fit(data, maxEpochs, batchSize, es)
        return self.denoiser.predict(data, batch_size=512)

    def transform(self, data):
        return self.denoiser.predict(data, batch_size=512)

class AE:
    def __init__(self, probModel, layerSize="auto", latentSize="auto"):
        if probModel == "NB":
            self.prob = nb_loglike
        if probModel == "Bernoulli":
            self.prob = bernoulli_loglike
        self.layerSize = layerSize
        self.latentSize = latentSize

    def _buildModels(self, data):
        self.probModel = self.prob(data.shape[1])
        layerSize = int(np.power(np.prod(data.shape),0.25)*4.0)
        latentSize = int(layerSize/4+1)
        sf = tf.keras.layers.Input((1,))
        countsIn = tf.keras.layers.Input((data.shape[1],))
        logCounts = self.probModel.inputTransform(countsIn/sf)
        logCounts = tf.keras.layers.GaussianDropout(0.05)(logCounts)
        e1 = tf.keras.layers.Dense(layerSize, "gelu")(logCounts)
        e2 = tf.keras.layers.Dense(layerSize, "gelu")(e1)
        e3 = tf.keras.layers.Dense(layerSize, "gelu")(e2)
        latent = tf.keras.layers.Dense(latentSize)(e3)
        reconstructed = tf.keras.layers.Dense(data.shape[1])(latent) + sf
        l = self.probModel([countsIn, reconstructed])
        self.model = tf.keras.Model([countsIn,sf], l)
        self.model.compile(tfa.optimizers.AdamW(1e-6, 1e-3, amsgrad=True), loss=lambda _, x:x)
        self.latentModel = tf.keras.Model([countsIn,sf], latent)
        self.denoiser = tf.keras.Model([countsIn,sf], self.probModel.linkFunction(reconstructed))
    
    def fit(self, data, sf, maxEpochs=500, batchSize="auto", es=10):
        self._buildModels(data)
        sfMat = sf[: ,None]
        es = tf.keras.callbacks.EarlyStopping("val_loss", patience=es, restore_best_weights=True)
        x_train, x_test, sf_train, sf_test = train_test_split(data, sfMat, test_size=0.05, random_state=42)
        if batchSize == "auto":
            batchSize = int(np.sqrt(len(data)))
        self.model.fit(x=(x_train,sf_train), y=np.zeros((len(x_train),1)), validation_data=((x_test,sf_test), np.zeros((len(x_test),1))),
                        epochs=maxEpochs, batch_size=batchSize, callbacks=[es], verbose=2)

    def fit_transform(self, data, sf, maxEpochs=500, batchSize="auto", es=10):
        self.fit(data, sf, maxEpochs, batchSize, es)
        return self.latentModel.predict((data,sf), batch_size=512)