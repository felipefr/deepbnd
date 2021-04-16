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

# Test Loading 

# sigma_prediction_ny40_mixed_200

NlistModel = [40]
predict_file = 'sigma_prediction_ny40_mixed_200.hd5'
predict_file_shear = 'sigma_prediction_ny{0}_mixed_eps2'

ns = 200
sigma = []
sigma_ref = []
error_eps1 = np.zeros((len(NlistModel), ns, 3))
error_eps2 = np.zeros((len(NlistModel), ns, 3))
error_rel_eps1 = np.zeros((len(NlistModel), ns, 3))
error_rel_eps2 = np.zeros((len(NlistModel), ns, 3))

for j, jj in enumerate(NlistModel): 
    print(myhd.loadhd5(predict_file.format(jj), 'error'))
    # error_eps2[j,:,:] =  myhd.loadhd5(predict_file_shear.format(jj), 'error')
    
    error_rel_eps1[j,:,:] =  myhd.loadhd5(predict_file.format(jj), 'error_rel')
    # error_rel_eps2[j,:,:] =  myhd.loadhd5(predict_file_shear.format(jj), 'error_rel')




# # ================ Alternative Prediction ========================================== 
folderImages = './images/'
suptitle = 'Error_Stress_vertical.{0}' 

plt.figure(1)
plt.title(suptitle.format('', ''))
plt.plot(NlistModel, np.mean(error_eps1[:,:,0], axis = 1), '-o', label = 'mean (DNN)')
plt.plot(NlistModel, np.mean(error_eps1[:,:,0], axis = 1) + np.std(error_eps1[:,:,0], axis = 1) , '--', label = 'mean + std (DNN)')
plt.plot(NlistModel, 3*[np.mean(error_eps1[0,:,1], axis = 0)], '-o', label = 'mean (Periodic)')
plt.plot(NlistModel, 3*[np.mean(error_eps1[0,:,2], axis = 0)], '-o', label = 'mean (Linear)')
# plt.ylim(1.0e-8,1.0e-5)
plt.yscale('log')
plt.xlabel('N')
plt.ylabel('Error homogenised stress Frobenius')
plt.grid()
plt.legend(loc = 'best')
plt.savefig(suptitle.format('png'))


folderImages = './images/'
suptitle = 'Error_relative_Stress_vertical.{0}' 

plt.figure(2)
plt.title(suptitle.format('', ''))
plt.plot(NlistModel, np.mean(error_rel_eps1[:,:,0], axis = 1), '-o', label = 'mean (DNN)')
plt.plot(NlistModel, np.mean(error_rel_eps1[:,:,0], axis = 1) + np.std(error_rel_eps1[:,:,0], axis = 1) , '--', label = 'mean + std (DNN)')
plt.plot(NlistModel, 3*[np.mean(error_rel_eps1[0,:,1], axis = 0)], '-o', label = 'mean (Periodic)')
plt.plot(NlistModel, 3*[np.mean(error_rel_eps1[0,:,2], axis = 0)], '-o', label = 'mean (Linear)')
# plt.ylim(1.0e-3,1.0e-1)
plt.yscale('log')
plt.xlabel('N')
plt.ylabel('Relative Error homogenised stress Frobenius')
plt.grid()
plt.legend(loc = 'best')
plt.savefig(suptitle.format('png'))

