import numpy as np
from sklearn import model_selection
from keras.layers import Input, Dense, Lambda
from keras.models import Model, load_model
from keras import backend as K
from keras import metrics
from keras import callbacks

#return prediction for x_test
def autoencoder_predict(encoder, x_test):
	prediction = encoder.predict(x_test)
	return prediction.reshape((len(prediction), np.prod(prediction.shape[1:])))

#return a fit deep encoder
def deep_autoencoder_fit(x_train, dimension=32, optimizer="adadelta", loss_function="binary_crossentropy", nb_epoch=2500, batch_size=100, save=0, suffix=""):
	input_img = Input(shape=(784,))

	x_train, x_valid = model_selection.train_test_split(x_train, test_size=0.10, random_state=0)

	# "encoded" is the encoded representation of the input
	encoded = Dense(1000, activation='relu', init="he_normal")(input_img)
	encoded = Dense(500, activation='relu', init="he_normal")(encoded)
	encoded = Dense(250, activation='relu', init="he_normal")(encoded)
	encoded = Dense(dimension, activation='linear', init="he_normal")(encoded)

	# "decoded" is the lossy reconstruction of the input
	decoded = Dense(250, activation='relu', init="he_normal")(encoded)
	decoded = Dense(500, activation='relu', init="he_normal")(decoded)
	decoded = Dense(1000, activation='relu', init="he_normal")(decoded)
	decoded = Dense(784, activation='sigmoid', init="he_normal")(decoded)

	autoencoder = Model(input=input_img, output=decoded)

	encoder = Model(input=input_img, output=encoded)

	autoencoder.compile(optimizer=optimizer, loss=loss_function)

	autoencoder.fit(x_train, x_train,
		nb_epoch=nb_epoch,
		verbose=0,
		batch_size=batch_size,
		shuffle=True,
		validation_data=(x_valid, x_valid),
		callbacks=[callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10),
		 callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)])
	
	if (save==1):
		encoder.save('./saved_models/deep_encoder'+suffix+'.h5')
	return encoder

#return a fit variational encoder
def variational_autoencoder_fit(x_train, dimension=32, optimizer="adadelta", nb_epoch=2500, batch_size=100, save=0, suffix=""):

	original_dim = 784
	latent_dim = dimension

	x_train, x_valid = model_selection.train_test_split(x_train, test_size=0.10, random_state=0)

	x = Input(batch_shape=(batch_size, original_dim))

	h = Dense(1000, activation='relu', init="he_normal")(x)
	h = Dense(500, activation='relu', init="he_normal")(h)
	h = Dense(250, activation='relu', init="he_normal")(h)
	h = Dense(latent_dim, activation='linear', init="he_normal")(h)

	z_mean = Dense(latent_dim)(h)
	z_log_var = Dense(latent_dim)(h)

	def sampling(args):
		z_mean, z_log_var = args
		epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
		return z_mean + K.exp(z_log_var / 2) * epsilon

	# note that "output_shape" isn't necessary with the TensorFlow backend
	z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

	# we instantiate these layers separately so as to reuse them later
	decoder_h = Dense(250, activation='relu', init="he_normal")(z)
	decoder_h = Dense(500, activation='relu', init="he_normal")(decoder_h)
	decoder_h = Dense(1000, activation='relu', init="he_normal")(decoder_h)
	x_decoded_mean = Dense(original_dim, activation='sigmoid', init="he_normal")(decoder_h)
	
	def vae_loss(x, x_decoded_mean):
		xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
		kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
		return xent_loss + kl_loss

	vae = Model(x, x_decoded_mean)
	vae.compile(optimizer=optimizer, loss=vae_loss)

	x_train, x_valid = model_selection.train_test_split(x_train, test_size=0.10, random_state=0)

	vae.fit(x_train, x_train,
			shuffle=True,
			verbose=0,
			nb_epoch=nb_epoch,
			batch_size=batch_size,
			validation_data=(x_valid, x_valid),
			callbacks=[callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10),
			 callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)])

	# build a model to project inputs on the latent space
	encoder = Model(x, z_mean)
	
	if (save==1):
		encoder.save('./saved_models/variational_encoder'+suffix+'.h5')
	return encoder