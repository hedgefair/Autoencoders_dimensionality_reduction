#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import csv
import time
from autoencoders import *
from sklearn import decomposition
from sklearn import datasets
from sklearn import metrics as metricsSklearn
from sklearn import neighbors
from sklearn import cluster
from sklearn import model_selection
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
from keras import backend as K
from keras import metrics

####################################
##           LOAD DATA            ##
####################################

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train.astype('float32') ) / 255.
x_test = (x_test.astype('float32') ) / 255.

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

x_train_convo = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test_convo = np.reshape(x_test, (len(x_test), 28, 28, 1))
   
####################################
##           FUNCTIONS            ##
####################################

# return fit pca
def pca_fit(x_train, dimension=32):
	return decomposition.PCA(n_components=dimension).fit(np.array(x_train))

# return prediction for x_test
def pca_predict(pca, x_test):
	return pca.transform(x_test)

# return fit lsa
def lsa_fit(x_train, dimension=32):
	return decomposition.TruncatedSVD(n_components=dimension).fit(np.array(x_train))

# return prediction for x_test
def lsa_predict(lsa, x_test):
	return lsa.transform(x_test)

def supervised_classifier_fit(x_train, y_train):
	return neighbors.KNeighborsClassifier(n_neighbors=10, n_jobs=-1).fit(x_train, y_train)

def classifier_predict(classifier, x_test):
	return  classifier.predict(x_test)

def precision(prediction, label):
	return float(np.sum((prediction-label) == 0))/len(label)

#Precision en fonction de la dimension de projection
def dimension_effect():
	print "Precision (%) de KNN en fonction de la dimension de projection"
	file = open("./results/dimension_results.csv", "wb")
	writer = csv.writer(file)
	writer.writerow(['Dimension',
		'Precision PCA','Train Time PCA','Predict Time PCA',
		'Precision LSA','Train Time LSA','Predict Time LSA',
		'Precision DAE','Train Time DAE','Predict Time DAE',
		'Precision VAE','Train Time VAE','Predict Time VAE',])
	for i in xrange(1,21):
		print "Dimension",i,"..."

		# PCA
		start = time.time()
		pca = pca_fit(x_train, dimension=i)
		time_pca_train = time.time() - start

		classifier_pca = supervised_classifier_fit(pca_predict(pca, x_train), y_train)

		start = time.time()
		predictions_pca = classifier_predict(classifier_pca, pca_predict(pca, x_test))
		time_pca_predict = time.time() - start

		# LSA
		start = time.time()
		lsa = lsa_fit(x_train, dimension=i)
		time_lsa_train = time.time() - start

		classifier_lsa = supervised_classifier_fit(lsa_predict(lsa, x_train), y_train)
		
		start = time.time()
		predictions_lsa = classifier_predict(classifier_lsa, lsa_predict(lsa, x_test))
		time_lsa_predict = time.time() - start

		# Deep Autoencoder
		start = time.time()
		deep_encoder = deep_autoencoder_fit(x_train, dimension=i, suffix="_dimension_"+str(i), save=1)
		time_dae_train = time.time() - start

		#deep_encoder = load_model('./saved_models/deep_encoder_dimension_'+str(i)+'.h5')
		classifier_dae = supervised_classifier_fit(autoencoder_predict(deep_encoder, x_train), y_train)
		
		start = time.time()
		predictions_dae = classifier_predict(classifier_dae, autoencoder_predict(deep_encoder, x_test))
		time_dae_predict = time.time() - start

		# Variational Autoencoder
		start = time.time()
		variational_encoder = variational_autoencoder_fit(x_train, dimension=i, suffix="_dimension_"+str(i), save=1)
		time_vae_train = time.time() - start

		#variational_encoder = load_model('./saved_models/variational_encoder_dimension_'+str(i)+'.h5')
		classifier_vae = supervised_classifier_fit(autoencoder_predict(variational_encoder, x_train), y_train)
		
		start = time.time()
		predictions_vae = classifier_predict(classifier_vae, autoencoder_predict(variational_encoder, x_test))
		time_vae_predict = time.time() - start

		writer.writerow([i,
			100*precision(predictions_pca, y_test), time_pca_train, time_pca_predict,
			100*precision(predictions_lsa, y_test), time_lsa_train, time_lsa_predict,
			100*precision(predictions_dae, y_test), time_dae_train, time_dae_predict,
			100*precision(predictions_vae, y_test), time_vae_train, time_vae_predict])
	file.close()

#Influence of train size over the precision of KNN
def trainsize_effect(iterations=5):
	print "Influence de la taille de l'ensemble d'apprentissage sur la precision de KNN"
	file = open("./results/trainsize_results.csv", "wb")
	writer = csv.writer(file)

	writer.writerow(['Taille',
		'Precision PCA','Train Time PCA','Predict Time PCA',
		'Precision LSA','Train Time LSA','Predict Time LSA',
		'Precision DAE','Train Time DAE','Predict Time DAE',
		'Precision VAE','Train Time VAE','Predict Time VAE',])
	for i in xrange(1,21,1):
		print "Taille ",i*60,"..."
		precision_pca = 0
		precision_lsa = 0
		precision_dae = 0
		precision_vae = 0
		time_pca_train = 0
		time_pca_predict = 0
		time_lsa_train = 0
		time_lsa_predict = 0
		time_dae_train = 0
		time_dae_predict = 0
		time_vae_train = 0
		time_vae_predict = 0

		for j in xrange(iterations):
			print "    Passe ",j+1,"..."
			# Create a reduced trainset
			_, x_train_reduced, _, y_train_reduced = model_selection.train_test_split(x_train, y_train, test_size=float(i)/1000, random_state=j)

			# PCA
			start = time.time()
			pca = pca_fit(x_train_reduced, dimension=5)
			time_pca_train = time_pca_train + time.time() - start

			classifier_pca = supervised_classifier_fit(pca_predict(pca, x_train_reduced), y_train_reduced)
			
			start = time.time()
			predictions_pca = classifier_predict(classifier_pca, pca_predict(pca, x_test))
			time_pca_predict = time_pca_predict + time.time() - start

			# LSA
			start = time.time()
			lsa = lsa_fit(x_train_reduced, dimension=5)
			time_lsa_train = time_lsa_train + time.time() - start

			classifier_lsa = supervised_classifier_fit(lsa_predict(lsa, x_train_reduced), y_train_reduced)

			start = time.time()
			predictions_lsa = classifier_predict(classifier_lsa, lsa_predict(lsa, x_test))
			time_lsa_predict = time_lsa_predict + time.time() - start

			# Deep Autoencoder
			start = time.time()
			deep_encoder = deep_autoencoder_fit(x_train_reduced, dimension=5, batch_size=len(x_train_reduced)/10)
			time_dae_train = time_dae_train + time.time() - start

			classifier_dae = supervised_classifier_fit(autoencoder_predict(deep_encoder, x_train_reduced), y_train_reduced)
			
			start = time.time()
			predictions_dae = classifier_predict(classifier_dae, autoencoder_predict(deep_encoder, x_test))
			time_dae_predict = time_dae_predict + time.time() - start


			# Variational Autoencoder
			start = time.time()
			variational_encoder = variational_autoencoder_fit(x_train_reduced, optimizer="adadelta", dimension=5, batch_size=1)
			time_vae_train = time_vae_train + time.time() - start

			classifier_vae = supervised_classifier_fit(autoencoder_predict(variational_encoder, x_train_reduced), y_train_reduced)
			
			start = time.time()
			predictions_vae = classifier_predict(classifier_vae, autoencoder_predict(variational_encoder, x_test))
			time_vae_predict = time_vae_predict + time.time() - start

			precision_pca = np.max([precision_pca, precision(predictions_pca, y_test)])
			precision_lsa = np.max([precision_lsa, precision(predictions_lsa, y_test)])
			precision_dae = np.max([precision_dae, precision(predictions_dae, y_test)])
			precision_vae = np.max([precision_vae, precision(predictions_vae, y_test)])

		writer.writerow([i*60, 100*precision_pca, time_pca_train, time_pca_predict,
		 100*precision_lsa, time_lsa_train/iterations, time_lsa_predict/iterations,
		 100*precision_dae, time_dae_train/iterations, time_dae_predict/iterations,
		 100*precision_vae, time_vae_train/iterations, time_vae_predict/iterations])
	file.close()

####################################
##              MAIN              ##
####################################

if __name__ == '__main__' :
	dimension_effect()
	trainsize_effect()