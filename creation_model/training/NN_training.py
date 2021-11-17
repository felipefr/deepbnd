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
from deepBND.creation_model.training.net_arch import standardNets
import deepBND.core.data_manipulation.utils as dman
import deepBND.core.data_manipulation.wrapper_h5py as myhd

def run_training(net):
    dman.exportScale(net.param['file_XY'], net.param['file_scaler'], nX, Nrb, Ylabel = 'Y_%s'%load_flag)
    scalerX, scalerY = dman.importScale(net.param['file_scaler'], nX, Nrb)
        
    net.param['Y_data_max'] = scalerY.data_max_ 
    net.param['Y_data_min'] = scalerY.data_min_
    net.param['X_data_max'] = scalerX.data_max_
    net.param['X_data_min'] = scalerX.data_min_
    net.param['scalerX'] = scalerX
    net.param['scalerY'] = scalerY
    
    dman.writeDict(net.getDict())
    
    XY_train = dman.getDatasetsXY(nX, Nrb, net.param['file_XY'], scalerX, scalerY, Ylabel = 'Y_%s'%load_flag)[0:2]
    XY_val = dman.getDatasetsXY(nX, Nrb, net.param['file_XY_val'], scalerX, scalerY, Ylabel = 'Y_%s'%load_flag)[0:2]
    
    hist = net.training(XY_train, XY_val)


if __name__ == '__main__':
    folderDataset = rootDataPath + "/deepBND/dataset/"
    folderTrain = rootDataPath + "/deepBND/training/"
    
    nameXY = folderDataset +  'XY.hd5'
    nameXY_val = folderDataset +  'XY_validation.hd5'

    if(len(sys.argv) > 1):    
        Nrb = int(sys.argv[1])
        epochs = int(sys.argv[2])
        archId = int(sys.argv[3])

    else:
        Nrb = 80
        epochs = 1000
        archId = 'small'
        load_flag = 'S'

    nX = 36
    
    print('Nrb is ', Nrb, 'epochs ', epochs)
    
    net = standardNets[archId]
    
    net.param['epochs'] = int(epochs)
    net.param['nY'] = Nrb
    net.param['nX'] = nX
    net.param['archId'] = archId
    net.param['stepEpochs'] = 1
    net.param['file_weights'] = folderTrain + 'models_weights_%s_%s_%d.hdf5'%(archId,load_flag,Nrb)
    net.param['file_net'] = folderTrain + 'models_net_%s_%s_%d.txt'%(archId,load_flag,Nrb)
    net.param['file_prediction'] = folderTrain + 'models_prediction_%s_%s_%d.txt'%(archId,load_flag,Nrb)
    net.param['file_scaler'] = folderTrain + 'scaler_%s_%d.txt'%(load_flag,Nrb)
    net.param['file_XY'] = nameXY
    net.param['file_XY_val'] = nameXY_val
    
    run_training(net)
    
    