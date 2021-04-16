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

folderTrain = './models/dataset_shear1/'
folderBasis = './models/dataset_shear1/'

NlistModel = [5,40,140]
predict_file = 'sigma_prediction_ny{0}.hd5'
predict_file_shear = 'sigma_prediction_ny{0}_shear.hd5'

ns = 100
sigma = []
sigma_ref = []
error = np.zeros((len(NlistModel), ns))
error_shear = np.zeros((len(NlistModel), ns))

for j, jj in enumerate(NlistModel): 
    error[j,:] =  myhd.loadhd5(predict_file.format(jj), 'error').flatten()**2
    error_shear[j,:] =  myhd.loadhd5(predict_file_shear.format(jj), 'error').flatten()**2



# # ================ Alternative Prediction ========================================== 
folderImages = './images/'
suptitle = 'Error_Stress_Pure_Axial_Shear_Square.{0}' 

plt.figure(1)
plt.title(suptitle.format('', ''))
plt.plot(NlistModel, np.mean(error, axis = 1), '-o', label = 'mean (axial)')
plt.plot(NlistModel, np.mean(error, axis = 1) + np.std(error, axis = 1) , '--', label = 'mean + std (axial)')
plt.plot(NlistModel, np.mean(error_shear, axis = 1), '-o', label = 'mean (shear)')
plt.plot(NlistModel, np.mean(error_shear, axis = 1) + np.std(error, axis = 1) , '--', label = 'mean + std (shear)')
# plt.ylim(1.0e-8,1.0e-5)
plt.yscale('log')
plt.xlabel('N')
plt.ylabel('Squared Error homogenised stress Frobenius')
plt.grid()
plt.legend(loc = 'best')
# plt.savefig(suptitle.format('png'))
