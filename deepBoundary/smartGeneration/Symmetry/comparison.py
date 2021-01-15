import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0,'../')
sys.path.insert(0,'../../../utils/')
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

import json
import copy

import matplotlib.pyplot as plt

f = open("../../../../rootDataPath.txt")
rootData = f.read()[:-1]
f.close()



nX = 36
nY = 40

folder = rootData + '/deepBoundary/smartGeneration/newTrainingSymmetry/'
folder2 = rootData + '/deepBoundary/smartGeneration/LHS_frozen_p4_volFraction/'
nameout = 'plot_history_LHS_p4_volFraction_drop02_nX{0}_nY{1}_{2}_history_.txt'
nameout_val = 'plot_history_LHS_p4_volFraction_drop02_nX{0}_nY{1}_{2}_history_val.txt'

eig = myhd.loadhd5(folder2 + 'eigens.hd5', 'eigenvalues')
errorPOD = np.zeros(160)

for i in range(160):
    errorPOD[i] = 0.1*np.sum(eig[i:])

hist = []
hist_val = []
for i in range(1,19):
    hist.append(np.loadtxt(folder + nameout.format(nX,nY,i) ))
    hist_val.append(np.loadtxt(folder + nameout_val.format(nX,nY,i) ))
    
lastError = []
lastError_val = []
for i in [7,8,9,10,11,12,13,1]:
    lastError.append(np.mean(hist[i][-5:]))
    lastError_val.append(np.mean(hist_val[i][-5:]))

lastError = np.array(lastError)
lastError_val = np.array(lastError_val)

Nlist = [5,10,15,20,25,30,35,40]
plt.figure(1)
plt.plot(Nlist,lastError, '-o', label = 'ErrorDNN_train')
plt.plot(Nlist,lastError_val, '-o', label = 'ErrorDNN_validation')
# plt.plot(np.arange(80),errorPOD[::2],'--', label = 'ErrorPOD')
plt.yscale('log')
plt.xlabel('N')
plt.ylabel('error')
plt.grid()
plt.legend()
plt.savefig('error_arch_mean_POD.png')

plt.figure(2)
plt.plot(errorPOD)
plt.yscale('log')

