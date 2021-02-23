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

def basicModelTraining(nsTrain, nX, nY, net, fnames):
    X, Y, scalerX, scalerY = syml.getTraining(0,nsTrain, nX, nY, fnames['file_X'], fnames['file_Y'])
    
    Neurons = net['Neurons']
    actLabel = net['activations']
    Epochs = net['epochs']
    decay = net['decay']
    lr = net['lr']
    saveFile = fnames['file_weights'] 
    stepEpochs = fnames['stepEpochs']
    ratio_val = 0.2
     
    indices = np.arange(len(X))
    np.random.seed(1)
    np.random.shuffle(indices)
    
    X = X[indices,:nX]
    Y = Y[indices,:nY]
    
    fac = 1.0

    maxAlpha = scalerY.data_max_
    minAlpha = scalerY.data_min_
    
    print(maxAlpha[0:5])
    print(minAlpha[0:5])
    input()
    
    w_l = fac*(maxAlpha - minAlpha)**2.0  
    w_l = w_l.astype('float32')

    model = mytf.DNNmodel_notFancy(nX, nY, net['Neurons'], net['activations'])
    
    num_parameters = 0 # in case of treating differently a part of the outputs
    history = mytf.my_train_model( model, X, Y, num_parameters, Epochs, lr = lr, decay = decay, 
                                  w_l = w_l, w_mu = 0.0, ratio_val = ratio_val,
                                  saveFile = saveFile.format(nY), stepEpochs = stepEpochs)
        
    mytf.plot_history( history, savefile = saveFile.format(nY)[:-5] + '_plot')
                
    return history

nX = 36 # because the only first 4 are relevant, the other are constant
nsBlock = 15360

Nrb = int(sys.argv[1])
epochs = int(sys.argv[2])

print('Nrb is ', Nrb, 'epochs ', epochs)

folder = './models/dataset_hybrid/'
nameYlist = folder +  'Y.h5'
nameEllipseData = folder + 'ellipseData.h5'

fnames = {}      
fnames['file_weights'] = './models/extendedSymmetry_newCase5/weights_ny{0}.hdf5'.format(Nrb)
fnames['file_X'] = nameEllipseData
fnames['file_Y'] = nameYlist
fnames['stepEpochs'] = 1


# series with 5000 epochs
net = {'Neurons': 5*[100], 'activations': ['relu','relu','sigmoid'], 'lr': 1.0e-4, 'decay' : 1.0} # normally reg = 1e-5
# net = {'Neurons': 5*[100], 'activations': ['tanh','relu','linear'], 'lr': 1.0e-4, 'decay' : 1.0} # normally reg = 1e-5
# net = {'Neurons': 4*[50], 'activations': ['tanh','relu','linear'], 'lr': 1.0e-4, 'decay' : 1.0} # normally reg = 1e-5

net['epochs'] = int(epochs)

start = timer()

hist = basicModelTraining(nsBlock, nX, Nrb, net, fnames)
end = timer()
