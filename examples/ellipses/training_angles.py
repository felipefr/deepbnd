#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:04:47 2022

@author: felipe
"""

"""
We aim to obtain model_weights.hd5, which storages the trained weights of NN models.
It should be run providing training and validation datasets.
Models should be trained indepedently for axial and shear loads ('A' and 'S' labels).
"""

import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np

from deepBND.__init__ import * 
import deepBND.creation_model.training.wrapper_tensorflow as mytf
# from deepBND.creation_model.training.net_arch import standardNets
from deepBND.creation_model.training.net_arch import NetArch
import deepBND.core.data_manipulation.utils as dman
import deepBND.core.data_manipulation.wrapper_h5py as myhd


# standardNets = {'big': NetArch([300, 300, 300], 3*['swish'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.005] + [0.0], 1.0e-8),
#          'small':NetArch([40, 40, 40], 3*['swish'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.005] + [0.0], 1.0e-8)}


# {'big': NetArch([300, 300, 300], 3*['swish'] + ['sigmoid'], 5.0e-4, 0.9, [0.0] + 3*[0.0] + [0.0], 0.0),
standardNets = {'big': NetArch([300, 300, 300], 3*['swish'] + ['sigmoid'], 5.0e-4, 0.9, [0.0] + 3*[0.0] + [0.0], 0.0),
         'small':NetArch([40, 40, 40], 3*['swish'] + ['linear'], 5.0e-4, 0.8, [0.0] + 3*[0.001] + [0.0], 1.0e-8),
         'big_classical': NetArch([300, 300, 300], 3*['swish'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.005] + [0.0], 1.0e-8) }


def dataAugmentation(XY):
    # the + 1 is applied in the very specific case of the scaling of sinus cosinus
    # 
    XX = np.concatenate((XY[0], -XY[0] + 1.0), axis = 0) 
    YY = np.concatenate((XY[1], XY[1]), axis = 0)
    
    ids = np.arange(0,XX.shape[0])
    np.random.shuffle(ids)
    

    return (XX[ids,:],YY[ids,:])

def toAngles(X):
    m = int(X.shape[1]/2)
    
    ang = np.zeros((X.shape[0], m))
    
    for i in range(X.shape[0]):
        for j in range(m):
            ang[i, j] = np.arctan2(X[i,2*j], X[i,2*j + 1])

    return ang

def run_training(net, Ylabel):
    dman.exportScale(net.files['XY'], net.files['scaler'], 2*net.nX, net.nY, Ylabel = Ylabel)
    scalerX, scalerY = dman.importScale(net.files['scaler'], 2*net.nX, net.nY)

    net.scalerX = scalerX
    net.scalerY = scalerY
    
    dman.writeDict(net.__dict__)
    
    XY_train = dman.getDatasetsXY(2*net.nX, net.nY, net.files['XY'], scalerX, scalerY, Ylabel = Ylabel)[0:2]
    XY_val = dman.getDatasetsXY(2*net.nX, net.nY, net.files['XY_val'], scalerX, scalerY, Ylabel = Ylabel)[0:2]
    
    
    X_train_original = scalerX.inverse_transform(XY_train[0])
    X_val_original = scalerX.inverse_transform(XY_val[0])
    
    X_train_ang = toAngles(X_train_original)
    X_val_ang = toAngles(X_val_original)    
    
    
    hist = net.training((X_train_ang, XY_train[1]), (X_val_ang, XY_val[1]))

    return XY_train, XY_val, scalerX, scalerY

if __name__ == '__main__':
    
    folderDataset = rootDataPath + "/ellipses/training/"
    folderTrain = rootDataPath + "/ellipses/training/"
    
    nameXY = folderDataset +  'XY_train.hd5'
    nameXY_val = folderDataset +  'XY_val.hd5'

    if(len(sys.argv) > 1):    
        Nrb = int(sys.argv[1])
        epochs = int(sys.argv[2])
        archId = int(sys.argv[3])

    else:
        Nrb = 80
        epochs = 100
        archId = 'big_classical'
        load_flag = 'S'

    nX = 36
    
    print('Nrb is ', Nrb, 'epochs ', epochs)
    
    net = standardNets[archId]
    
    suffix = "angles"
    
    net.epochs =  int(epochs)
    net.nY = Nrb
    net.nX = nX
    net.archId =  archId
    net.stepEpochs =  1
    net.files['weights'] = folderTrain + 'models_weights_%s_%s_%d_%s.hdf5'%(archId,load_flag,Nrb,suffix)
    net.files['net_settings'] =  folderTrain + 'models_net_%s_%s_%d.txt'%(archId,load_flag,Nrb)
    net.files['prediction'] = folderTrain + 'models_prediction_%s_%s_%d.txt'%(archId,load_flag,Nrb)
    net.files['scaler'] = folderTrain + 'scaler_%s_%d.txt'%(load_flag,Nrb)
    net.files['XY'] = nameXY
    net.files['XY_val'] = nameXY_val
    
    XY_train, XY_val, scalerX, scalerY = run_training(net, 'Y_%s'%load_flag)
    
    
