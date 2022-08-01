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
         'big_classical': NetArch([300, 300, 300], 3*['swish'] + ['linear'], 5.0e-3, 0.1, [0.0] + 3*[0.005] + [0.0], 1.0e-8),
         'big_big': NetArch([300, 300, 300, 300], 4*['swish'] + ['linear'], 5.0e-3, 0.1, [0.0] + 4*[0.005] + [0.0], 1.0e-8)}


def dataAugmentation(XY):
    XX = np.concatenate((XY[0], -XY[0]), axis = 0)
    YY = np.concatenate((XY[1], XY[1]), axis = 0)
    
    ids = np.arange(0,XX.shape[0])
    np.random.shuffle(ids)
    

    return (XX[ids,:],YY[ids,:])


def run_training(net, Ylabel):
    dman.exportScale(net.files['XY'], net.files['scaler'], net.nX, net.nY, Ylabel = Ylabel)
    scalerX, scalerY = dman.importScale(net.files['scaler'], nX, Nrb)

    net.scalerX = scalerX
    net.scalerY = scalerY
    
    dman.writeDict(net.__dict__)
    
    XY_train = dman.getDatasetsXY(nX, Nrb, net.files['XY'], scalerX, scalerY, Ylabel = Ylabel)[0:2]
    XY_val = dman.getDatasetsXY(nX, Nrb, net.files['XY_val'], scalerX, scalerY, Ylabel = Ylabel)[0:2]
    
    
    X_train_original = scalerX.inverse_transform(XY_train[0])
    X_val_original = scalerX.inverse_transform(XY_val[0])
    
    XY_train = (X_train_original, XY_train[1])
    XY_val = (X_val_original, XY_val[1])
    
    # XY_train = dataAugmentation(XY_train)
    # XY_val = dataAugmentation(XY_val)
    
    hist = net.training(XY_train, XY_val)

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
        Nrb = 140
        epochs = 500
        archId = 'big_classical'
        load_flag = 'A'

    nX = 72
    
    print('Nrb is ', Nrb, 'epochs ', epochs)
    
    net = standardNets[archId]
    
    suffix = "unnormalising_lr5em3"
    
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
    
    
