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

f = open("../../../rootDataPath.txt")
rootData = f.read()[:-1]
f.close()

folderDataset = rootData + "/new_fe2/dataset/"
folderTrain = rootData + "/new_fe2/training/"

nameXY = folderDataset +  'XY_train.hd5'
nameXY_val = folderDataset +  'XY_validation.hd5'

Nrb = 80
epochs = 1000
archId = 'small'
load_flag = 'S'


# Nrb = int(sys.argv[1])
# epochs = int(sys.argv[2])
# archId = int(sys.argv[3])
nX = 36

print('Nrb is ', Nrb, 'epochs ', epochs)

nets = {}
# nets['big'] = {'Neurons': [300, 300, 300], 'activations': 3*['swish'] + ['linear'], 'lr': 5.0e-4, 'decay' : 0.1, 'drps' : [0.0] + 3*[0.005] + [0.0], 'reg' : 1.0e-8}
# nets['small'] = {'Neurons': [40, 40, 40], 'activations': 3*['swish'] + ['linear'], 'lr': 5.0e-4, 'decay' : 0.1, 'drps' : [0.0] + 3*[0.005] + [0.0], 'reg' : 1.0e-8}

nets['big'] = {'Neurons': [300, 300, 300], 'activations': 3*['swish'] + ['linear'], 'lr': 5.0e-4, 'decay' : 0.1, 'drps' : [0.0] + 3*[0.005] + [0.0], 'reg' : 1.0e-8}
nets['small'] = {'Neurons': [40, 40, 40], 'activations': 3*['swish'] + ['linear'], 'lr': 5.0e-4, 'decay' : 0.1, 'drps' : [0.0] + 3*[0.005] + [0.0], 'reg' : 1.0e-8}


net = nets[archId]

net['epochs'] = int(epochs)
net['nY'] = Nrb
net['nX'] = nX
net['archId'] = archId
net['nsTrain'] = 5*10240 
net['nsVal'] = 10240
net['stepEpochs'] = 1
net['file_weights'] = folderTrain + 'models/weights_%s_%s_%d.hdf5'%(archId,load_flag,Nrb)
net['file_net'] = folderTrain + 'models/net_%s_%s_%d.txt'%(archId,load_flag,Nrb)
net['file_prediction'] = folderTrain + 'models/prediction_%s_%s_%d.txt'%(archId,load_flag,Nrb)
net['file_scaler'] = folderTrain + 'scaler_%s_%d.txt'%(load_flag,Nrb)
net['file_XY'] = [nameXY, nameXY_val]

exportScale(net['file_XY'][0], net['file_scaler'], nX, Nrb, Ylabel = 'Y_%s'%load_flag)
scalerX, scalerY = importScale(net['file_scaler'], nX, Nrb)
    
net['Y_data_max'] = scalerY.data_max_ 
net['Y_data_min'] = scalerY.data_min_
net['X_data_max'] = scalerX.data_max_
net['X_data_min'] = scalerX.data_min_
net['scalerX'] = scalerX
net['scalerY'] = scalerY
net['routine'] = 'generalModel_dropReg'

writeDict(net)

XY_train = getDatasetsXY(nX, Nrb, net['file_XY'][0], scalerX, scalerY, Ylabel = 'Y_%s'%load_flag)[0:2]
XY_val = getDatasetsXY(nX, Nrb, net['file_XY'][1], scalerX, scalerY, Ylabel = 'Y_%s'%load_flag)[0:2]

model = generalModel_dropReg(nX, Nrb, net)    
model.summary()

start = timer()
hist = basicModelTraining(XY_train, XY_val, model, net)
end = timer()