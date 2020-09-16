import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0,'../')
sys.path.insert(0,'../../utils/')
import matplotlib.pyplot as plt
import numpy as np

import myTensorflow as mytf
import tensorflow as tf
from timeit import default_timer as timer

import h5py
import pickle
import Generator as gene
import myHDF5 
import dataManipMisc as dman 
from sklearn.preprocessing import MinMaxScaler
import miscPosprocessingTools as mpt
import myHDF5 as myhd
from functools import partial

import json

def getLoadfunc(namefile, label):
       
    loadfunc = np.loadtxt
    if(namefile[-3:]=='hd5'):
        loadfunc = partial(myhd.loadhd5, label = label)
            
    return loadfunc

myloadfile = lambda x, y : getLoadfunc(x,y)(x)


def getTraining(ns_start, ns_end, nX, nY, Xdatafile, Ydatafile, scalerX = None, scalerY = None):
    X = np.zeros((ns_end - ns_start,nX))
    Y = np.zeros((ns_end - ns_start,nY))
 
        
    loadfunc = getLoadfunc(Ydatafile, 'Ylist')
    
    for i in range(ns_end - ns_start):
        j = i + ns_start
        X[i,:] = np.loadtxt(Xdatafile.format(j))[:nX,2]
    
    Y = loadfunc(Ydatafile)[ns_start:ns_end,:nY] 
   

    if(type(scalerX) == type(None)):
        scalerX = MinMaxScaler()
        scalerX.fit(X)
    
    if(type(scalerY) == type(None)):
        scalerY = MinMaxScaler()
        scalerY.fit(Y)
            
    return scalerX.transform(X), scalerY.transform(Y), scalerX, scalerY


def basicModelTraining(nsTrain, nX, nY, net, fnames):
    X, Y, scalerX, scalerY = getTraining(0,nsTrain, nX, nY, fnames['prefix_in_X'], fnames['prefix_in_Y'])
    
    Neurons = net['Neurons']
    actLabel = net['activations']
    drps = net['drps']
    reg = net['reg']
    Epochs = net['epochs']
    decay = net['decay']
    lr = net['lr']
    
    model = mytf.DNNmodel(nX, nY, Neurons, actLabel = actLabel , drps = drps, lambReg = reg  )
    
    num_parameters = 0 # in case of treating differently a part of the outputs
    history = mytf.my_train_model( model, X, Y, num_parameters, Epochs, lr = lr, decay = decay, w_l = 1.0, w_mu = 0.0)
        
    mytf.plot_history( history, savefile = fnames['prefix_out'] + 'plot_history' + fnames['suffix_out'])
    
    with open(fnames['prefix_out'] + 'history' + fnames['suffix_out'] + '.dat', 'wb') as f:
        pickle.dump(history.history, f)
        
    model.save_weights( fnames['prefix_out'] + 'weights' + fnames['suffix_out'])

    return history, model