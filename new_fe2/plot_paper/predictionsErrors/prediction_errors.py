import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0,'../utils/')
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

import tensorflow_for_training as tfut


rootDataPath = open('../../../rootDataPath.txt','r').readline()[:-1]

loadType = 'S'

folderModels = rootDataPath + '/new_fe2/training/models/'
folderScaler = rootDataPath + '/new_fe2/training/scalers/'
folderData = rootDataPath + '/new_fe2/dataset/'


nameXY = folderData +  'XY_train.hd5'


nameXY_val = folderData +  'XY_validation.hd5'

# Nrb = int(sys.argv[1])
Nrb = 5
archId = 'big'
epochs = 1
nX = 36
 

nameScaleXY = folderScaler +  'scaler_%s_%d.txt'%(loadType,Nrb)


net = {'Neurons': 3*[300], 'activations': 3*['swish'] + ['linear'], 'lr': 5.0e-4, 'decay' : 0.1, 'drps' : [0.0] + 3*[0.005] + [0.0], 'reg' : 1.0e-8}

net['epochs'] = int(epochs)
net['nY'] = Nrb
net['nX'] = nX
net['archId'] = archId
net['nsTrain'] = int(51200) 
net['nsVal'] = int(10240)
net['stepEpochs'] = 1
net['file_weights'] = folderModels + 'weights_%s_%s_%d.hdf5'%(archId,loadType,Nrb)
net['file_net'] = folderModels + 'net_%s_%s_%d.txt'%(archId,loadType,Nrb)
net['file_prediction'] = 'prediction_%s_%d.txt'%(loadType,Nrb)
net['file_XY'] = [nameXY, nameXY_val]
net['routine'] = 'generalModel_dropReg'

scalerX, scalerY  = tfut.importScale(nameScaleXY, nX, Nrb)

XY_train = tfut.getDatasetsXY(nX, Nrb, net['file_XY'][0], scalerX, scalerY,  Ylabel = 'Y_%s'%loadType)[0:2]
XY_val = tfut.getDatasetsXY(nX, Nrb, net['file_XY'][1], scalerX, scalerY,  Ylabel = 'Y_%s'%loadType)[0:2]

# Prediction 
X, Y = tfut.getDatasetsXY(nX, Nrb, net['file_XY'], scalerX, scalerY, 'Y_%s'%loadType)[0:2]
w_l = (scalerY.data_max_ - scalerY.data_min_)**2.0

nsTrain = net['nsTrain']
X_scaled = []; Y_scaled = []
X_scaled.append(X[:nsTrain,:]); Y_scaled.append(Y[:nsTrain,:])
X_scaled.append(X[nsTrain:,:]); Y_scaled.append(Y[nsTrain:,:])

nameXYlist = [folderData +  'XY_test.hd5']

for nameXY in nameXYlist:
    Xtemp, Ytemp = tfut.getDatasetsXY(nX, Nrb, nameXY, scalerX, scalerY, Ylabel = 'Y_%s'%loadType)[0:2]
    X_scaled.append(Xtemp); Y_scaled.append(Ytemp)

netp = net
modelp = tfut.generalModel_dropReg(nX, Nrb, netp)   
oldWeights = netp['file_weights']

modelp.load_weights(oldWeights)

error_stats = []
for Xi, Yi in zip(X_scaled,Y_scaled): 
    Yi_p = modelp.predict(Xi)
    error = tf.reduce_sum(tf.multiply(w_l,tf.square(tf.subtract(Yi_p,Yi))), axis=1).numpy()
    
    error_stats.append(np.array([np.mean(error),np.std(error), np.max(error), np.min(error)]))

np.savetxt(net['file_prediction'],np.array(error_stats))
