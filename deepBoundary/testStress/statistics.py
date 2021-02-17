import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0,'../')
sys.path.insert(0,'../../utils/')

import generation_deepBoundary_lib as gdb
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
import meshUtils as meut

import json
import copy

import matplotlib.pyplot as plt
import symmetryLib as syml

f = open("../../../rootDataPath.txt")
rootData = f.read()[:-1]
f.close()

# Test Loading 

ntest  = 5120
folderTest = rootData + '/deepBoundary/testStress/'
nameYtest = folderTest + 'Y_all.h5'
nameEllipseDataTest = folderTest + 'ellipseData_17.h5'
nameSnapsTest = folderTest + 'snapshots_17_all.h5'


Ytest = myhd.loadhd5(nameYtest, 'Ylist')
ellipseDataTest = myhd.loadhd5(nameEllipseDataTest, 'ellipseData')


# Model Prediction
nX = 36
nY = 140
ns = 10240

folderModels = rootData + '/deepBoundary/smartGeneration/newTrainingSymmetry/'
folderTrain = rootData + "/deepBoundary/smartGeneration/LHS_frozen_p4_volFraction/"
nameEllipseDataTrain = folderTrain + "ellipseData_1.h5"
nameYtrain = folderTrain + "Y.h5"
nameSnapsTrain = folderTrain + 'snapshots_all.h5'

folderTrainFS = rootData + '/deepBoundary/smartGeneration/LHS_p4_fullSymmetric/'
nameSnapsFS = folderTrain + 'snapshots_all.h5'

Ytrain = myhd.loadhd5(nameYtrain, 'Ylist')

IsolT, fIsolT = myhd.loadhd5_openFile(nameSnapsTest,['a', 'B','sigma','sigmaTotal'], mode = 'r')
Isol_a_T, Isol_B_T, sigma_T, sigmaTotal_T = IsolT

Isol, fIsol = myhd.loadhd5_openFile(nameSnapsTrain,['a', 'B','sigma'], mode = 'r')
Isol_a, Isol_B, sigma = Isol

for i in range(3):
    print('sigma train {0} = {1:e} +- {2:e} $'.format(['11','22','12'][i], np.mean(sigma[:,i]),np.std(sigma[:,i])))
    print('sigma test {0} = {1:e} +- {2:e} '.format(['11','22','12'][i], np.mean(sigma_T[:,i]),np.std(sigma_T[:,i])))


fIsol.close()
fIsolT.close()

plt.figure(1)
# plt.plot(np.mean(Ytest,axis=0))
plt.xlabel('N')
plt.ylabel('Y')
plt.plot(np.abs(np.mean(Ytest,axis=0)), label = 'test')
plt.plot(np.abs(np.mean(Ytest,axis=0)) + np.std(Ytest,axis=0), '--', label = 'test + std')
plt.plot(np.abs(np.mean(Ytest,axis=0)) - np.std(Ytest,axis=0), '--', label = 'test - std')

# plt.plot(np.mean(Ytrain,axis=0))
plt.plot(np.abs(np.mean(Ytrain,axis=0)), label = 'train')
plt.plot(np.abs(np.mean(Ytrain,axis=0)) + np.std(Ytrain,axis=0), '--', label = 'train + std')
plt.plot(np.abs(np.mean(Ytrain,axis=0)) - np.std(Ytrain,axis=0), '--', label = 'train + std')

plt.yscale('log')
plt.legend(loc= 'best')

plt.show()