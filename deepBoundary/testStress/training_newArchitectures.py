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
import tensorflow as tf

from tensorflow_for_training import *


folder = './models/dataset_new5/'
nameXY = folder +  'XY.h5'

folderVal = './models/dataset_newTest3/'
nameXY_val = folderVal +  'XY_Wbasis5.h5'


Nrb = int(sys.argv[1])
epochs = int(sys.argv[2])
archId = int(sys.argv[3])
nX = 36

print('Nrb is ', Nrb, 'epochs ', epochs)

# net = {'Neurons': [40], 'activations': 3*['swish'], 'lr': 1.0e-3, 'decay' : 1.0,
#         'drps' : 3*[0.0], 'reg' : 0.0}

# net = {'Neurons': [1000,1000,1000], 'activations': 4*['swish'], 'lr': 5.0e-4, 'decay' : 1.0, 'drps' : [0.0] + 3*[0.005] + [0.0], 'reg' : 1.0e-8}

net = {'Neurons': [10, 10, 10], 'activations': 3*['swish'] + ['linear'], 'lr': 5.0e-4, 'decay' : 1.0, 'drps' : [0.0] + 3*[0.005] + [0.0], 'reg' : 1.0e-8}

net['epochs'] = int(epochs)
net['nY'] = Nrb
net['nX'] = nX
net['archId'] = archId
net['nsTrain'] = int(51200) 
net['nsVal'] = int(5120)
net['stepEpochs'] = 1
net['file_weights'] = './models/newArchitectures/new5itself/weights_ny{0}_arch{1}_newLR.hdf5'.format(Nrb,archId)
net['file_net'] = './models/newArchitectures/new5itself/net_ny{0}_arch{1}.txt'.format(Nrb,archId)
net['file_prediction'] = './models/newArchitectures/new5itself/prediction_ny{0}_arch{1}.txt'.format(Nrb,archId)
net['file_XY'] = [nameXY, nameXY_val]

scalerX, scalerY = syml.getDatasetsXY(nX, Nrb, net['file_XY'][0])[2:4]

net['Y_data_max'] = scalerY.data_max_ 
net['Y_data_min'] = scalerY.data_min_
net['X_data_max'] = scalerX.data_max_
net['X_data_min'] = scalerX.data_min_
net['scalerX'] = scalerX
net['scalerY'] = scalerY
net['routine'] = 'generalModel_dropReg'

writeDict(net)

XY_train = syml.getDatasetsXY(nX, Nrb, net['file_XY'][0], scalerX, scalerY)[0:2]
XY_val = syml.getDatasetsXY(nX, Nrb, net['file_XY'][1], scalerX, scalerY)[0:2]


model = generalModel_dropReg(nX, Nrb, net)    
model.summary()


start = timer()
hist = basicModelTraining(XY_train, XY_val, model, net)
end = timer()

# oldWeights = './models/newArchitectures/new4/weights_ny{0}_arch{1}_retaken6.hdf5'.format(Nrb,archId)
# model.load_weights(oldWeights)


# Prediction 
X, Y = syml.getDatasetsXY(nX, Nrb, net['file_XY'], scalerX, scalerY)[0:2]
w_l = (scalerY.data_max_ - scalerY.data_min_)**2.0

nsTrain = net['nsTrain']
X_scaled = []; Y_scaled = []
X_scaled.append(X[:nsTrain,:]); Y_scaled.append(Y[:nsTrain,:])
X_scaled.append(X[nsTrain:,:]); Y_scaled.append(Y[nsTrain:,:])


nameXYlist = ['./models/dataset_new4/XY_Wbasis5.h5','./models/dataset_newTest2/XY_Wbasis5.h5','./models/dataset_test/XY_Wbasis5.h5']

for nameXY in nameXYlist:
    Xtemp, Ytemp = syml.getDatasetsXY(nX, Nrb, nameXY, scalerX, scalerY)[0:2]
    X_scaled.append(Xtemp); Y_scaled.append(Ytemp)

netp = net
modelp = generalModel_dropReg(nX, Nrb, netp)   
oldWeights = netp['file_weights']

modelp.load_weights(oldWeights)

error_stats = []
for Xi, Yi in zip(X_scaled,Y_scaled): 
    Yi_p = modelp.predict(Xi)
    error = tf.reduce_sum(tf.multiply(w_l,tf.square(tf.subtract(Yi_p,Yi))), axis=1).numpy()
    
    error_stats.append(np.array([np.mean(error),np.std(error), np.max(error), np.min(error)]))

np.savetxt(net['file_prediction'],np.array(error_stats))
