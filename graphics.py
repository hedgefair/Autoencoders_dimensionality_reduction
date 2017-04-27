#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from script import *
from autoencoders import *
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

def plot_results(encoded_imgs,tabn):
    x = encoded_imgs[:,0]
    y = encoded_imgs[:,1]
    xx = x[tabn]
    yy = y[tabn]
    cc = y_test[tabn]
    colors = cm.nipy_spectral(np.linspace(0,9,10)/10)
    plt.figure(figsize=(10,10))
    for i in xrange(0,10):
        plt.scatter(xx[cc==i], yy[cc==i], color=colors[i],label=i, s=3)
    plt.legend()

####################################
##              MAIN              ##
####################################

if __name__ == '__main__' :
    
    tabn = xrange(10000)

    #### Plot the projection into a plan of the test set
    #### Deep autoencoder
    encoder = load_model('./autoencoders/deep_encoder_projection_dimension_2.h5')
    X = encoder.predict(x_test)
    plot_results(X,tabn=tabn)
    plt.title(u'Projection of MNIST test set with a deep autoencoder')
    plt.show()
    #### Variational autoencoder
    encoder = load_model('./autoencoders/variational_encoder_projection_dimension_2.h5')
    X = encoder.predict(x_test)
    plot_results(X,tabn=tabn)
    plt.title(u'Projection of MNIST test set with a variational autoencoder')
    plt.show()
    #### PCA
    pca = pca_fit(x_train, dimension=2)
    X  = pca.transform(x_test)
    plot_results(X,tabn=tabn)
    plt.title(u'Projection of MNIST test set with PCA')
    plt.show()
    #### LSA
    lsa = lsa_fit(x_train, dimension=2)
    X  = lsa.transform(x_test)
    plot_results(X,tabn=tabn)
    plt.title(u'Projection of MNIST test set with LSA')
    plt.show()

    #### Observing the gaussian distribution of a variational autoencoder :
    #### ie the effect of the Kullback-Leibler regularization term.
    vae = load_model('./autoencoders/variational_encoder_dimension_4.h5')
    Z = vae.predict(x_test)
    k = len(Z[1,:])
    f, axs = plt.subplots(1, k, sharey=True)
    f.set_size_inches((16,6))
    for i in range(k):
        ax = axs[i]
        x = Z[:,i] 
        n, bins, patches = ax.hist(x, 50, normed=1, facecolor='blue', alpha=0.75,stacked=True)
        y = mlab.normpdf( bins, 0, 1)
        l = ax.plot(bins, y, 'r--', linewidth=1)
        ax.set_xlabel('Z[:,%i]'%i)
        ax.set_ylabel('Probability')
        #ax.set_title(r'$\mathrm{Histogram\ of\ Z[:,%i]:}$'%i)
    plt.legend(['N(0,1)'])
    f.suptitle(u'$\mathrm{Histogrammes\ des\ parties\ encodées\ pour\ un\ VAE\ à\ 4\ dimensions\ en\ comparaison\ avec\ la\ densité\ normale}$')
    plt.show()

    #### Observing the percentage of variance explained by each components of PCA
    pca_full = pca_fit(x_train, dimension=784)
    var_n = pca_full.explained_variance_
    # Variance cumulée
    plt.bar(range(1,51),np.cumsum(var_n[0:50]/np.sum(var_n)), color='green', alpha=0.75)
    # Variance par dimension
    plt.bar(range(1,51),var_n[0:50]/np.sum(var_n), color='blue', alpha=0.75)
    plt.legend([u'Variance expliquée',u'Variance expliquée cumulée'])
    plt.title(u"Analyse de la variance expliquée par chaque composante principale de l'ACP (Données d'entrainement du MNIST)")
    