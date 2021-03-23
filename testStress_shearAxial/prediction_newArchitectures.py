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
# import symmetryLib as syml
import tensorflow as tf

from tensorflow_for_training import *

def exportScale(filenameIn, filenameOut, nX, nY):
    scalerX, scalerY = syml.getDatasetsXY(nX, nY, filenameIn)[2:4]
    scalerLimits = np.zeros((max(nX,nY),4))
    scalerLimits[:nX,0] = scalerX.data_min_
    scalerLimits[:nX,1] = scalerX.data_max_
    scalerLimits[:nY,2] = scalerY.data_min_
    scalerLimits[:nY,3] = scalerY.data_max_

    np.savetxt(filenameOut, scalerLimits)

def importScale(filenameIn, nX, nY):
    scalerX = myMinMaxScaler()
    scalerY = myMinMaxScaler()
    scalerX.fit_limits(np.loadtxt(filenameIn)[:,0:2])
    scalerY.fit_limits(np.loadtxt(filenameIn)[:,2:4])
    scalerX.set_n(nX)
    scalerY.set_n(Nrb)
    
    return scalerX, scalerY


folder = './models/dataset_axial1/'
nameXY = folder +  'XY.h5'
nameScaleXY = folder +  'scaler.txt'
exportScale(nameXY, nameScaleXY, 36, 160)

folderVal = './models/dataset_axial2/'
nameXY_val = folderVal +  'XY_Wbasis1.h5'

# Nrb = int(sys.argv[1])
Nrb = int(input("Nrb="))
archId = 1
epochs = 1
nX = 36
# 
net = {'Neurons': 3*[300], 'activations': 3*['swish'] + ['linear'], 'lr': 5.0e-4, 'decay' : 0.1, 'drps' : [0.0] + 3*[0.005] + [0.0], 'reg' : 1.0e-8}

net['epochs'] = int(epochs)
net['nY'] = Nrb
net['nX'] = nX
net['archId'] = archId
net['nsTrain'] = int(51200) 
net['nsVal'] = int(5120)
net['stepEpochs'] = 1
net['file_weights'] = folder + 'models/weights_ny{0}_arch{1}.hdf5'.format(Nrb,archId)
net['file_net'] = folder + 'models/net_ny{0}_arch{1}.txt'.format(Nrb,archId)
net['file_prediction'] = folder + 'models/prediction_ny{0}_arch{1}.txt'.format(Nrb,archId)
net['file_XY'] = [nameXY, nameXY_val]
net['routine'] = 'generalModel_dropReg'

scalerX, scalerY  = importScale(nameScaleXY, nX, Nrb)

XY_train = syml.getDatasetsXY(nX, Nrb, net['file_XY'][0], scalerX, scalerY)[0:2]
XY_val = syml.getDatasetsXY(nX, Nrb, net['file_XY'][1], scalerX, scalerY)[0:2]

# Prediction 
X, Y = syml.getDatasetsXY(nX, Nrb, net['file_XY'], scalerX, scalerY)[0:2]
w_l = (scalerY.data_max_ - scalerY.data_min_)**2.0

nsTrain = net['nsTrain']
X_scaled = []; Y_scaled = []
X_scaled.append(X[:nsTrain,:]); Y_scaled.append(Y[:nsTrain,:])
X_scaled.append(X[nsTrain:,:]); Y_scaled.append(Y[nsTrain:,:])

nameXYlist = ['./models/dataset_axial3/XY_Wbasis1.h5']

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
