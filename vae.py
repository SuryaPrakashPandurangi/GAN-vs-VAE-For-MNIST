import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import vaeConfig


activation = vaeConfig.activation
padding = vaeConfig.padding

def plot(encoder, decoder):
    n = 30
    digitSize = 28
    figure = np.zeros((digitSize * n, digitSize * n))

    scale = 2.0
    figsize = 15
    xgrid = np.linspace(-scale, scale, n)
    y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(y):
        for j, xi in enumerate(xgrid):
            sam = np.array([[xi, yi]])
            x_decoded = decoder.predict(sam)
            digit = x_decoded[0].reshape(digitSize, digitSize)
            figure[
                i * digitSize : (i + 1) * digitSize,
                j * digitSize : (j + 1) * digitSize,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    beginning = digitSize // 2
    end = n * digitSize + beginning + 1
    pixel_range = np.arange(beginning, end, digitSize)
    xSamples = np.round(xgrid, 1)
    ySamples = np.round(y, 1)
    plt.xticks(pixel_range, xSamples)
    plt.yticks(pixel_range, ySamples)
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.imshow(figure, cmap=vaeConfig.cmap)
    plt.show()

class Sampling(layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


latentDimentionality = 2

def returnMNISTData():
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    return (x_train, _), (x_test, _)

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder(data)
            reconstruction = decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 28 * 28
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "Loss": total_loss,
            "Reconstruction Loss": reconstruction_loss,
            "KL Loss": kl_loss,
        }

def encoder():
    x = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, activation=activation, strides=2, padding=padding)(x)
    x = layers.Conv2D(64, 3, activation=activation, strides=2, padding=padding)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation=activation)(x)
    z_mean = layers.Dense(latentDimentionality, name="z_mean")(x)
    z_log_var = layers.Dense(latentDimentionality, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(x, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    return encoder

def decoder():
    x = keras.Input(shape=(latentDimentionality,))
    x = layers.Dense(7 * 7 * 64, activation=activation)(x)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation=activation, strides=2, padding=padding)(x)
    x = layers.Conv2DTranspose(32, 3, activation=activation, strides=2, padding=padding)(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation=vaeConfig.activationSigmoid,padding=padding)(x)
    decoder = keras.Model(latentDimentionality, decoder_outputs, name=vaeConfig.decoderName)
    decoder.summary()
    return decoder


(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

mnistData = np.concatenate([x_train, x_test], axis=0)
mnistData = np.expand_dims(mnistData, -1).astype("float32") / 255

encoder = encoder()
decoder = decoder()

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(),metrics=metrics)
vae.fit(mnistData, epochs=500, batch_size=128)

plot(encoder, decoder)
