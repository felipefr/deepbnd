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
    errorPOD[i] = np.sum(eig[i:])

hist_1 = []
hist_val_1 = []

hist_2 = []
hist_val_2 = []

hist_3 = []
hist_val_3 = []

hist_4 = []
hist_val_4 = []

hist_5 = []
hist_val_5 = []

for i in [15,16,17,18]:    
    hist_1.append(np.loadtxt(folder + nameout.format(nX,nY,i) ))
    hist_val_1.append(np.loadtxt(folder + nameout_val.format(nX,nY,i) ))

for i in [19,20,21,22,23]:    
    hist_2.append(np.loadtxt(folder + nameout.format(nX,nY,i) ))
    hist_val_2.append(np.loadtxt(folder + nameout_val.format(nX,nY,i) ))

# for i in [26,24]:    
#     hist_3.append(np.loadtxt(folder + nameout.format(nX,nY,i) ))
#     hist_val_3.append(np.loadtxt(folder + nameout_val.format(nX,nY,i) ))

# for i in [27,25]:    
#     hist_4.append(np.loadtxt(folder + nameout.format(nX,nY,i) ))
#     hist_val_4.append(np.loadtxt(folder + nameout_val.format(nX,nY,i) ))

for i in [29,28]:    
    hist_3.append(np.loadtxt(folder + nameout.format(nX,nY,i) ))
    hist_val_3.append(np.loadtxt(folder + nameout_val.format(nX,nY,i) ))

for i in [8,9,10,11,12,13,14,1]:
    hist_4.append(np.loadtxt(folder + nameout.format(nX,nY,i) ))
    hist_val_4.append(np.loadtxt(folder + nameout_val.format(nX,nY,i) ))
    
for i in [35,36,37,38,39,40,41,42,31,32,33,34]:
    hist_5.append(np.loadtxt(folder + nameout.format(nX,140,i) ))
    hist_val_5.append(np.loadtxt(folder + nameout_val.format(nX,140,i) ))

Nlist_1_2 = [5,15,25,35,40]
Nlist_3 = [5,40]
Nlist_4 = [5,10,15,20,25,30,35,40]
Nlist_5 = [5,10,15,20,25,30,35,40,80,100,120,140]
    
lastError_1 = []
lastError_val_1 = []

lastError_2 = []
lastError_val_2 = []

lastError_3 = []
lastError_val_3 = []

lastError_4 = []
lastError_val_4 = []

lastError_5 = []
lastError_val_5 = []

for i in range(len(hist_1)):
    lastError_1.append(np.mean(hist_1[i][-5:]))
    lastError_val_1.append(np.mean(hist_val_1[i][-5:]))

for i in range(len(hist_2)):
    lastError_2.append(np.mean(hist_2[i][-5:]))
    lastError_val_2.append(np.mean(hist_val_2[i][-5:]))

for i in range(len(hist_3)):
    lastError_3.append(np.mean(hist_3[i][-5:]))
    lastError_val_3.append(np.mean(hist_val_3[i][-5:]))

for i in range(len(hist_4)):
    lastError_4.append(np.mean(hist_4[i][-5:]))
    lastError_val_4.append(np.mean(hist_val_4[i][-5:]))

for i in range(len(hist_5)):
    lastError_5.append(np.mean(hist_5[i][-5:]))
    lastError_val_5.append(np.mean(hist_val_5[i][-5:]))

lastError_1 = np.array(lastError_1)
lastError_val_1 = np.array(lastError_val_1)

lastError_2 = np.array(lastError_2)
lastError_val_2 = np.array(lastError_val_2)

lastError_3 = np.array(lastError_3)
lastError_val_3 = np.array(lastError_val_3)

lastError_4 = np.array(lastError_4)
lastError_val_4 = np.array(lastError_val_4)

lastError_5 = np.array(lastError_5)
lastError_val_5 = np.array(lastError_val_5)

plt.figure(1)
# plt.plot(Nlist_1_2[:-1],lastError_1, '-o', label = 'ErrorDNN_train 1')
# plt.plot(Nlist_1_2,lastError_2, '-o', label = 'ErrorDNN_train 2')
# plt.plot(Nlist_3,lastError_3, '-o', label = 'ErrorDNN_train 3')
# plt.plot(Nlist_4,lastError_4, '-o', label = 'ErrorDNN_train 4')
# plt.plot(Nlist_5,lastError_5, '-o', label = 'ErrorDNN_train 5')

plt.plot(Nlist_1_2[:-1],lastError_val_1, '-o', label = 'ErrorDNN_val 1')
plt.plot(Nlist_1_2,lastError_val_2, '-o', label = 'ErrorDNN_val 2')
plt.plot(Nlist_3,lastError_val_3, '-o', label = 'ErrorDNN_val 3')
plt.plot(Nlist_4,lastError_val_4, '-o', label = 'ErrorDNN_val 4')
plt.plot(Nlist_5,lastError_val_5, '-o', label = 'ErrorDNN_val 5')

# plt.plot(Nlist,lastError_val, '-o', label = 'ErrorDNN_validation')
plt.plot(np.arange(160),errorPOD,'--', label = 'ErrorPOD')
# plt.plot(Nlist,errorPOD[Nlist] + lastError, '-o', label = 'Total Error')
plt.yscale('log')
plt.xlabel('N')
plt.ylabel('error')
plt.grid()
plt.legend()
plt.savefig('error_complete_shear_severalArchs_val.png')

plt.figure(2)
plt.plot(errorPOD)
plt.yscale('log')

