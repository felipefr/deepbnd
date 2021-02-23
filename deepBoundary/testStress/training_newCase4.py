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

def basicModelTraining(nsTrain, nsVal, nX, nY, net, fnames):
    X, Y, scalerX, scalerY = syml.getDatasets(nX, nY, fnames['file_X'], fnames['file_Y'])
    
    Neurons = net['Neurons']
    actLabel = net['activations']
    Epochs = net['epochs']
    decay = net['decay']
    lr = net['lr']
    saveFile = fnames['file_weights'] 
    stepEpochs = fnames['stepEpochs']
     
    np.random.seed(1)
    indices = np.arange(0,nsTrain)
    np.random.shuffle(indices)
    
    Xtrain = X[indices,:nX]
    Ytrain = Y[indices,:nY]

    np.random.seed(2)
    indices = np.arange(nsTrain, len(X))
    np.random.shuffle(indices)

    Xval = X[indices[:nsVal],:nX]
    Yval = Y[indices[:nsVal],:nY]
    
    fac = 1.0

    maxAlpha = scalerY.data_max_
    minAlpha = scalerY.data_min_
    
    print(scalerY.data_max_[0:5])
    print(scalerY.data_min_[0:5])
    
    w_l = fac*(maxAlpha - minAlpha)**2.0  
    w_l = w_l.astype('float32')

    model = mytf.DNNmodel_notFancy(nX, nY, net['Neurons'], net['activations'])
    
    num_parameters = 0 # in case of treating differently a part of the outputs
    history = mytf.my_train_model( model, Xtrain, Ytrain, num_parameters, Epochs, lr = lr, decay = decay, 
                                  w_l = w_l, w_mu = 0.0, saveFile = saveFile, 
                                  stepEpochs = stepEpochs, validationSet = {'X':Xval, 'Y': Yval})
        
    mytf.plot_history( history, savefile = saveFile[:-5] + '_plot')
                
    return history

nX = 36 # because the only first 4 are relevant, the other are constant
nsBlock = 10240
nsTrain = nsBlock
nsVal = int(nsBlock/4) # half of the test set

Nrb = int(sys.argv[1])
epochs = int(sys.argv[2])

print('Nrb is ', Nrb, 'epochs ', epochs)

folder = './models/dataset_extendedSymmetry_recompute/'
nameYlist = folder +  'Y_fourth.h5'
nameEllipseData = folder + 'ellipseData_fourth.h5'

folderVal = './models/dataset_test/'
nameYlist_val = folderVal +  'Y_extended.h5'
nameEllipseData_val = folderVal + 'ellipseData.h5'

if(Nrb<0):
    print("just create new dataset")
    nameYlist_origin = folder +  'Y.h5'
    nameEllipseData_origin = folder + 'ellipseData.h5'

    ind = np.arange(0,nsBlock)
    
    myhd.savehd5(nameYlist,myhd.loadhd5(nameYlist_origin,'Ylist')[ind,:],'Ylist', mode = 'w')
    myhd.savehd5(nameEllipseData,myhd.loadhd5(nameEllipseData_origin,'ellipseData')[ind,:,:], 'ellipseData', mode = 'w')

    sys.exit() 

fnames = {}      
fnames['file_weights'] = './models/extendedSymmetry_newCase4/weights_ny{0}_linear_lrm4_small.hdf5'.format(Nrb)
fnames['file_X'] = [nameEllipseData,nameEllipseData_val]
fnames['file_Y'] = [nameYlist, nameYlist_val]
fnames['stepEpochs'] = 1



# series with 5000 epochs
# net = {'Neurons': 5*[100], 'activations': ['relu','relu','sigmoid'], 'lr': 1.0e-4, 'decay' : 1.0} # normally reg = 1e-5
# net = {'Neurons': 5*[100], 'activations': ['tanh','relu','linear'], 'lr': 1.0e-4, 'decay' : 1.0} # normally reg = 1e-5
net = {'Neurons': 4*[50], 'activations': ['tanh','relu','linear'], 'lr': 1.0e-4, 'decay' : 1.0} # normally reg = 1e-5

net['epochs'] = int(epochs)

start = timer()

hist = basicModelTraining(nsTrain, nsVal, nX, Nrb, net, fnames)
end = timer()
