import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0,'../utils/')

# import generation_deepBoundary_lib as gdb
import matplotlib.pyplot as plt
import numpy as np

import myTensorflow as mytf
from timeit import default_timer as timer

import h5py
import pickle
import myHDF5 
import dataManipMisc as dman 
from sklearn.preprocessing import MinMaxScaler
import miscPosprocessingTools as mpt
import myHDF5 as myhd

import matplotlib.pyplot as plt
import symmetryLib as syml
import tensorflow as tf


normStress = lambda s : np.sqrt(s[0]**2 + s[1]**2 + 2*s[2]**2)
reduceNormStress = lambda S: np.array([normStress(S[i,:]) for i in range(len(S))])

# Test Loading 

typeModel = 'shear'

NlistModel = [5,40,140]
predict_file = 'sigma_prediction_ny{0}_{1}.hd5'

NlistArch_direct = [1,2,3] 
predict_file_direct = 'sigma_prediction_{0}_stress_direct_arch{1}.hd5'

ns = 100
sigma = []
sigma_ref = []
error = np.zeros((len(NlistModel), ns))
sigma = np.zeros((len(NlistModel), ns, 3))
sigma_ref = np.zeros((len(NlistModel), ns, 3))

error_direct = np.zeros((len(NlistArch_direct), ns))

randIndices = np.loadtxt("randomIndicesTest_axial.txt").astype('int') # same as shear

for j, jj in enumerate(NlistModel): 
    temp = myhd.loadhd5(predict_file.format(jj,typeModel), 'sigma_ref')
    print(temp.shape)
    sigma_ref[j,:,:] =  myhd.loadhd5(predict_file.format(jj,typeModel), 'sigma_ref')
    sigma[j,:,:] =  myhd.loadhd5(predict_file.format(jj,typeModel), 'sigma')
    error[j,:] =  reduceNormStress(sigma[j,:,:] - sigma_ref[j,:,:])/reduceNormStress(sigma_ref[j,:,:])

for j, jj in enumerate(NlistArch_direct): 
    error_direct[j,:] =  myhd.loadhd5(predict_file_direct.format(typeModel,jj), 'error').flatten()[randIndices]/reduceNormStress(sigma_ref[j,:,:])



# # ================ Alternative Prediction ========================================== 
folderImages = './images/'
suptitle = 'Relative_Error_Std_{0}_vs_direct.{1}'.format(typeModel,'') 

plt.figure(1)
plt.title(suptitle.format('', ''))
plt.plot(NlistModel, np.mean(error, axis = 1), '-o', label = 'mean ({0} bndmethod)'.format(typeModel))
plt.plot(NlistModel, np.mean(error, axis = 1) + np.std(error, axis = 1) , '--', label = 'mean + std ({0} bndmethod)'.format(typeModel))
# for i in NlistArch_direct:
for i in [1]:
    plt.plot([0,140], 2*[np.mean(error_direct[i-1,:])] , '-', label = 'mean ({0} direct arch{1})'.format(typeModel,i))
    plt.plot([0,140], 2*[np.mean(error_direct[i-1,:]) + np.std(error_direct[i-1,:])] , '--', label = 'mean + std ({0} direct arch{1})'.format(typeModel,i))

# plt.ylim(1.0e-8,1.0e-5)
plt.yscale('log')
plt.xlabel('N')
plt.ylabel('Relative Error homogenised stress Frobenius')
plt.grid()
plt.legend(loc = 'best')
plt.savefig(suptitle.format(typeModel,'png'))
