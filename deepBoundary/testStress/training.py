import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0,'../')
sys.path.insert(0,'../../utils/')
import matplotlib.pyplot as plt
import numpy as np

import myTensorflow as mytf
from timeit import default_timer as timer

import h5py
import pickle
# import Generator as gene
import myHDF5 
import dataManipMisc as dman 
from sklearn.preprocessing import MinMaxScaler
import miscPosprocessingTools as mpt
import myHDF5 as myhd
import symmetryLib as syml

import json
import copy

def basicModelTraining(nsTrain, nX, nY, net, fnames, w_l = 1.0):
    X, Y, scalerX, scalerY = syml.getTraining(0,nsTrain, nX, nY, fnames['prefix_in_X'], fnames['prefix_in_Y'])
    
    Neurons = net['Neurons']
    actLabel = net['activations']
    Epochs = net['epochs']
    decay = net['decay']
    lr = net['lr']; weightThreshold = net['weightTsh']
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    X = X[indices,:]
    Y = Y[indices,:]
    
    fac = 1.0e3
    
    if(w_l<0.0): # to indicate we want to compute problem-specific weights
        maxAlpha = scalerY.data_max_
        minAlpha = scalerY.data_min_
        
        w_l = fac*(maxAlpha - minAlpha)**2.0
        w_l[weightThreshold:] = 0.0      
        w_l = w_l.astype('float32')

    model = mytf.DNNmodel_notFancy(nX, nY, net['Neurons'], net['activations'])
    
    num_parameters = 0 # in case of treating differently a part of the outputs
    history = mytf.my_train_model( model, X, Y, num_parameters, Epochs, lr = lr, decay = decay, w_l = w_l, w_mu = 0.0)
        
    mytf.plot_history( history, savefile = fnames['prefix_out'] + 'plot_history' + fnames['suffix_out'])
            
    model.save_weights(fnames['prefix_out'] + fnames['suffix_out'])
    # model.save(fnames['prefix_out'] + fnames['suffix_out'])
    
    return history

Nrb = int(sys.argv[1])
epochs = int(sys.argv[2])

print('Nrb is ', Nrb, 'epochs ', epochs)

folder = './models/dataset_extendedSymmetry_recompute/'
nameYlist = folder +  'Y.h5'
nameEllipseData = folder + 'ellipseData.h5'

fnames = {}      
fnames['suffix_out'] = 'weights_ny{0}'.format(Nrb)
fnames['prefix_out'] = './models/extendedSymmetry_new/'
fnames['prefix_in_X'] = nameEllipseData
fnames['prefix_in_Y'] = nameYlist


nX = 36 # because the only first 4 are relevant, the other are constant
nY = 160 # 10 alphas
nsTrain = 4*10240

# series with 5000 epochs
net = {'Neurons': 3*[50], 'activations': ['relu','relu','sigmoid'], 'lr': 1.0e-4, 'decay' : 1.0} # normally reg = 1e-5

net['epochs'] = int(epochs)
net['weightTsh'] = int(Nrb)
#os.system("mkdir " + fnames['prefix_out']) # in case the folder is not present

start = timer()

hist = basicModelTraining(nsTrain, nX, nY, net, fnames, w_l = -1.0)
end = timer()

plt.figure(2)
plt.plot(hist.history['mse'])
plt.plot(hist.history['val_mse'])
plt.grid()
plt.yscale('log')
plt.savefig(fnames['prefix_out'] + '/plot_mse_{0}.png'.format(Nrb))
#plt.show()
