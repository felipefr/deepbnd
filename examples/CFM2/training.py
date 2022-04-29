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


standardNets = {'huge': NetArch([500, 500, 500], 3*['swish'] + ['linear'], 5.0e-4, 0.1, 5*[0.0], 1.0e-8),
                'huge_tanh': NetArch([500, 500, 500], 3*['swish'] + ['tanh'], 5.0e-4, 0.1, 5*[0.0], 1.0e-8),
                'huge_reg': NetArch([500, 500, 500], 3*['swish'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.005] + [0.0], 1.0e-9),
                'escalonated': NetArch([50, 300, 500], 3*['swish'] + ['linear'], 5.0e-4, 0.1, 5*[0.0], 1.0e-9),
                'big': NetArch([300, 300, 300], 3*['swish'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.005] + [0.0], 1.0e-8),
                'extreme_big_noreg': NetArch([500, 500, 500], 3*['swish'] + ['linear'], 5.0e-4, 0.1, 2*[0.0] + 2*[0.05] + [0.0], 0.0),
                'extreme_medbig_noreg': NetArch([400, 400, 400], 3*['swish'] + ['linear'], 5.0e-4, 0.1, 5*[0.0], 0.0),
                'extreme_medbig_reg': NetArch([400, 400, 400], 3*['swish'] + ['linear'], 5.0e-3, 0.1, 5*[0.0], 0.0),
                'extreme_big_noreg2': NetArch([400, 400, 400, 400], 4*['swish'] + ['linear'], 5.0e-4, 0.1, 6*[0.0], 0.0),
                'big_noreg': NetArch([300, 300, 300], 3*['swish'] + ['linear'], 1.0e-3, 0.1, 5*[0.0], 0.0),
                'medium_noreg': NetArch([200, 200, 200], 3*['swish'] + ['linear'], 5.0e-4, 0.1, 5*[0.0], 0.0),
                'medium': NetArch([100, 100, 100], 3*['swish'] + ['linear'], 5.0e-4, 0.1, 5*[0.0], 1.0e-8),
                'small':NetArch([40, 40, 40], 3*['tanh'] + ['linear'], 1.0e-5, 0.1, 5*[0.0], 0.0)}


def run_training(net, Ylabel, Xmask = None):
    print(Ylabel)
    dman.exportScale(net.files['XY'], net.files['scaler'], net.nX, net.nY, Ylabel = Ylabel, scalerType = 'MinMax11' )
    scalerX, scalerY = dman.importScale(net.files['scaler'], nX, Nrb, scalerType = 'MinMax11')

    net.scalerX = scalerX
    net.scalerY = scalerY
    
    dman.writeDict(net.__dict__)
    
    XY_train = dman.getDatasetsXY(nX, Nrb, net.files['XY'], scalerX, scalerY, Ylabel = Ylabel)[0:2]
    XY_val = dman.getDatasetsXY(nX, Nrb, net.files['XY_val'], scalerX, scalerY, Ylabel = Ylabel)[0:2]
        
    # if(type(Xmask) != type(None)):
    #     net.nX = len(Xmask)
    #     XY_train = (XY_train[0][:, Xmask] , XY_train[1])
    #     XY_val = (XY_val[0][:, Xmask] , XY_val[1])
    
    
    # if(type(Xmask) != type(None)):
    #     lacking = np.array(list(set(list(np.arange(36))) - set(list(Xmask))))  
    #     XY_train[0][:, lacking] = 0.0
    #     XY_val[0][:, lacking] = 0.0
      
    
    hist = net.training(XY_train, XY_val, seed = 2)

    return XY_train, XY_val, scalerX, scalerY

if __name__ == '__main__':
    
    folderDataset = rootDataPath + "/CFM2/datasets_fluctuations/"
    folderTrain = rootDataPath + "/CFM2/training_fluctuations_normalisation/"
    
    nameXY = folderDataset +  'XY_train.hd5'
    nameXY_val = folderDataset +  'XY_validation.hd5'

    if(len(sys.argv) > 1):    
        Nrb = int(sys.argv[1])
        epochs = int(sys.argv[2])
        archId = int(sys.argv[3])

    else:
        Nrb = 140
        epochs = 100
        archId = 'huge'
        load_flag = 'A'
        suffix = "all"

    nX = 36
    
    print('Nrb is ', Nrb, 'epochs ', epochs)
    
    net = standardNets[archId]

    Xmask_list = {'all' : np.arange(nX), 
                  '35' : np.arange(nX - 6),
                  '4x4' : np.array([7,8,9,10,13,14,15,16,19,20,21,22,25,26,27,28]),
                  '2x2' : np.array([14,15,20,21]), 
                  '4x4_nobottom' : np.array([13,14,15,16,19,20,21,22,25,26,27,28]),
                  '4x4_nobottom' : np.array([13,14,15,16,19,20,21,22,25,26,27,28])}
    
    net.epochs =  int(epochs)
    net.nY = Nrb
    net.nX = nX
    net.archId =  archId
    net.stepEpochs =  1
    net.files['weights'] = folderTrain + 'model_weights_%s_%s_%d_%s.hdf5'%(archId,load_flag,Nrb,suffix)
    net.files['net_settings'] =  folderTrain + 'model_net_%s_%s_%d_%s.txt'%(archId,load_flag,Nrb,suffix)
    net.files['prediction'] = folderTrain + 'model_prediction_%s_%s_%d_%s.txt'%(archId,load_flag,Nrb,suffix)
    net.files['scaler'] = folderTrain + 'scalers_%s_%s.txt'%(load_flag, suffix)
    net.files['XY'] = nameXY
    net.files['XY_val'] = nameXY_val
    net.files["tensorboard_id"] = "%s_%s_%d_%s"%(archId, load_flag, Nrb, suffix)
    
    XY_train, XY_val, scalerX, scalerY = run_training(net, 'Y_%s'%load_flag, Xmask_list[suffix])
    
    
