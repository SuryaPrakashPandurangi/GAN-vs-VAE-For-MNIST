import keras
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack

from numpy.random import randn
from numpy.random import randint

from keras.optimizers import Adam
from keras.models import Sequential

from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten

from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU

from keras.layers import Dropout

from matplotlib import pyplot


ganDirectory = r'.\ganplots'

def returnAdamOptimizer():
	return Adam(lr=0.0002, beta_1=0.5)

def returnDiscriminator(shape=(28,28,1)):
	padding = 'same'
	sig = 'sigmoid'
	lossEnt ='binary_crossentropy'
	model = Sequential()

    model.add(Conv2D(64, (3,3), strides=(2, 2), padding=padding, input_shape=shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))

    model.add(Conv2D(64, (3,3), strides=(2, 2), padding=padding))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(1, activation=sig))

	model.compile(loss=lossEnt, optimizer=returnAdamOptimizer(), metrics=['accuracy'])
	return model



def returnGenerator(dims):
	model = Sequential()

	nodes = 128 * 7 * 7

    model.add(Dense(nodes, input_dim=dims))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))

    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
	return model


def returnGAN(gen, dis):
	dis.trainable = False
	model = Sequential()
	model.add(gen)
	model.add(dis)

	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

def loadMNISTData():

	(trainX, _), (testX, _) = keras.datasets.mnist.load_data()

	newtrainX = []
	for i in range(5000):
		newtrainX.append(trainX[i])

	print("Size of : "+ str(len(trainX)))

	X = expand_dims(trainX, axis=-1)
	X = X.astype('float32')
	X = X / 255.0
	return X

def generateRealDataSamples(dataset, sampleCount):
	ix = randint(0, dataset.shape[0], sampleCount)
	X = dataset[ix]
	y = ones((sampleCount, 1))
	return X, y


def generateFakeData(generator, latentDims, samples):
	x = getLatentPoints(latentDims, samples)
	X = generator.predict(x)
	y = zeros((samples, 1))
	return X, y


def getLatentPoints(dims, sampleSize):
	x = randn(dims * sampleSize)
	x = x.reshape(sampleSize, dims)
	return x


def plotResults(examples, epoch, n=10):
	for i in range(n * n):
		pyplot.subplot(n, n, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	filename = ganDirectory +"\\"+'generated_plot_e%03d.png' % (epoch+1)
	pyplot.savefig(filename)
	pyplot.close()

def summarize(epoch, generator, discriminator, dataset, latent_dim, sampleCount=100):
	X_real, y_real = generate_real_samples(dataset, sampleCount)
	_, acc_real = discriminator.evaluate(X_real, y_real, verbose=0)
	x_fake, y_fake = generateFakeData(generator, latent_dim, sampleCount)
	_, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
	plotResults(x_fake, epoch)

def train(generator, discriminator, gan, dataset, latent, epochs=500, batch=128):
	batchesPerEpoch = int(dataset.shape[0] / batch)
	halfBatch = int(batch / 2)

	for i in range(epochs):
        for j in range(batchesPerEpoch):
			X_real, y_real = generateRealDataSamples(dataset, halfBatch)
			X_fake, y_fake = generateFakeData(generator, latent, halfBatch)
			X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
			discLoss, _ = discriminator.train_on_batch(X, y)
			X_gan = getLatentPoints(latent, batch)
			y_gan = ones((nbatch, 1))
			genLoss = gan.train_on_batch(X_gan, y_gan)
			print('>Epoch: %d, Discriminator loss: %.3f, Generator Loss=%.3f' % (i+1, discriminator, genLoss))

		summarize(i, generator, d_model, dataset, latent_dim)


latentSpaceDimensions = 100

dataset = loadMNISTData()

discriminator = returnDiscriminator()
generator = returnGenerator(latentSpaceDimensions)
gan = returnGAN(generator, discriminator)

train(generator, discriminator, gan, dataset, latentSpaceDimensions)
