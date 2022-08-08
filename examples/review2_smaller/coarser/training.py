#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 17:36:27 2022

@author: felipe
"""

"""
We aim to obtain model_weights.hd5, which storages the trained weights of NN models.
It should be run providing training and validation datasets.
Models should be trained indepedently for axial and shear loads ('A' and 'S' labels).
"""

import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['HDF5_DISABLE_VERSION_CHECK']='2'

import matplotlib.pyplot as plt
import numpy as np

from deepBND.__init__ import * 
import deepBND.creation_model.training.wrapper_tensorflow as mytf
# from deepBND.creation_model.training.net_arch import standardNets
from deepBND.creation_model.training.net_arch import NetArch
import deepBND.core.data_manipulation.utils as dman
import deepBND.core.data_manipulation.wrapper_h5py as myhd


standardNets = {'big_tri': NetArch([150, 300, 500], 3*['swish'] + ['linear'], 5.0e-4, 0.1, 2*[0.0] + [0.005] + 2*[0.0], 1.0e-8),
                'big': NetArch([300, 300, 300], 3*['swish'] + ['linear'], 5.0e-4, 0.1, [0.0] + 1*[0.005] + [0.0], 1.0e-8),
                'small':NetArch([40, 40, 40], 3*['swish'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.005] + [0.0], 1.0e-8)}


# {'big': NetArch([300, 300, 300], 3*['swish'] + ['sigmoid'], 5.0e-4, 0.9, [0.0] + 3*[0.0] + [0.0], 0.0),
# standardNets = {'big': NetArch([300, 300, 300], 3*['swish'] + ['sigmoid'], 5.0e-4, 0.9, [0.0] + 3*[0.0] + [0.0], 0.0),
#          'small':NetArch([40, 40, 40], 3*['swish'] + ['linear'], 5.0e-4, 0.8, [0.0] + 3*[0.001] + [0.0], 1.0e-8),
#          'big_classical': NetArch([300, 300, 300], 3*['swish'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.005] + [0.0], 1.0e-8),
#          'big_big_400': NetArch([400, 400, 400], 3*['swish'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.005] + [0.0], 1.0e-8)}


def run_training(net, Ylabel):
    # dman.exportScale(net.files['XY'], net.files['scaler'], net.nX, net.nY, Ylabel = Ylabel, scalerType = 'MinMax11' )
    scalerX, scalerY = dman.importScale(net.files['scaler'], nX, Nrb, scalerType = 'MinMax11')

    net.scalerX = scalerX
    net.scalerY = scalerY
    
    dman.writeDict(net.__dict__)
    
    XY_train = dman.getDatasetsXY(nX, Nrb, net.files['XY'], scalerX, scalerY, Ylabel = Ylabel)[0:2]
    XY_val = dman.getDatasetsXY(nX, Nrb, net.files['XY_val'], scalerX, scalerY, Ylabel = Ylabel)[0:2]
    
    hist = net.training(XY_train, XY_val, seed = 2)

    return XY_train, XY_val, scalerX, scalerY


if __name__ == '__main__':
    
    folderTrain = rootDataPath + "/review2_smaller/training_coarser/"
    
    suffix = 'translation'
    nameXY = folderTrain +  'XY_%s_train.hd5'%suffix
    nameXY_val = folderTrain +  'XY_%s_val.hd5'%suffix

    if(len(sys.argv) > 1):    
        Nrb = int(sys.argv[1])
        epochs = int(sys.argv[2])
        archId = int(sys.argv[3])

    else:
        Nrb = 160
        epochs = 5000
        archId = 'big'
        load_flag = 'S'

    nX = 72
    
    print('Nrb is ', Nrb, 'epochs ', epochs)
    
    net = standardNets[archId]
    
    net.epochs =  int(epochs)
    net.nY = Nrb
    net.nX = nX
    net.archId =  archId
    net.stepEpochs =  1
    net.files['weights'] = folderTrain + 'models_weights_%s_%s_%d_%s.hdf5'%(archId,load_flag,Nrb,suffix)
    net.files['net_settings'] =  folderTrain + 'models_net_%s_%s_%d.txt'%(archId,load_flag,Nrb)
    net.files['prediction'] = folderTrain + 'models_prediction_%s_%s_%d.txt'%(archId,load_flag,Nrb)
    net.files['scaler'] = folderTrain + 'scaler_%s_%s.txt'%(suffix,load_flag)
    net.files['XY'] = nameXY
    net.files['XY_val'] = nameXY_val
    
    XY_train, XY_val, scalerX, scalerY = run_training(net, 'Y_%s'%load_flag)
    
    
