import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
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
import tensorflow as tf

from tensorflow_for_training import *


folder = './models/dataset_partially_axial1/'
nameXY = folder +  'XY.h5'

folderVal = './models/dataset_partially_axial2/'
nameXY_val = folderVal +  'XY.h5'

Nrb = 40
epochs = 500
archId = 1


# Nrb = int(sys.argv[1])
# epochs = int(sys.argv[2])
# archId = int(sys.argv[3])
nX = 36

print('Nrb is ', Nrb, 'epochs ', epochs)


netBig = {'Neurons': [300, 300, 300], 'activations': 3*['swish'] + ['linear'], 'lr': 5.0e-4, 'decay' : 0.1, 'drps' : [0.0] + 3*[0.005] + [0.0], 'reg' : 1.0e-8}
netSmall = {'Neurons': [40, 40, 40], 'activations': 3*['swish'] + ['linear'], 'lr': 5.0e-4, 'decay' : 0.1, 'drps' : [0.0] + 3*[0.005] + [0.0], 'reg' : 1.0e-8}

net = netSmall

net['epochs'] = int(epochs)
net['nY'] = Nrb
net['nX'] = nX
net['archId'] = archId
net['nsTrain'] = int(5*10240) 
net['nsVal'] = int(5120)
net['stepEpochs'] = 1
net['file_weights'] = './models/dataset_partially_axial1/models/weights_small.hdf5'
net['file_net'] = './models/dataset_partially_axial1/models/net_small.txt'
net['file_prediction'] = './models/dataset_partially_axial1/models/prediction_small.txt'
net['file_scaler'] = './models/dataset_partially_axial1/models/scaler.txt'
net['file_XY'] = [nameXY, nameXY_val]

exportScale(net['file_XY'][0], net['file_scaler'], nX, Nrb)
scalerX, scalerY = importScale(net['file_scaler'], nX, Nrb)
    
net['Y_data_max'] = scalerY.data_max_ 
net['Y_data_min'] = scalerY.data_min_
net['X_data_max'] = scalerX.data_max_
net['X_data_min'] = scalerX.data_min_
net['scalerX'] = scalerX
net['scalerY'] = scalerY
net['routine'] = 'generalModel_dropReg'

writeDict(net)

XY_train = getDatasetsXY(nX, Nrb, net['file_XY'][0], scalerX, scalerY)[0:2]
XY_val = getDatasetsXY(nX, Nrb, net['file_XY'][1], scalerX, scalerY)[0:2]

model = generalModel_dropReg(nX, Nrb, net)    
model.summary()

start = timer()
hist = basicModelTraining(XY_train, XY_val, model, net)
end = timer()