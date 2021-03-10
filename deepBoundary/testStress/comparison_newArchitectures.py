import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0,'../')
sys.path.insert(0,'../../utils/')

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

folderTrain = './models/dataset_new5/'
folderBasis = './models/dataset_new4/'

NlistModel = [5,10,20,40,80,140]
predict_file = './models/newArchitectures_cluster/new5/prediction_ny{0}_arch{1}.txt'


errors = np.zeros((2,6,5,4)) # arch, Nrb, Dataset, Stat

for i in range(2):
    for j in range(6): 
        errors[i,j,:,:] = np.loadtxt(predict_file.format(NlistModel[j],i+1))



# # ======================= Reading theoretical POD =================== 
nsTrain = 51200
nbasis = 160
nameEigen = folderBasis + 'eigens.hd5'

eig = myhd.loadhd5(nameEigen, 'eigenvalues')   
errorPOD = np.zeros(nbasis)

for i in range(nbasis):
    errorPOD[i] = np.sum(eig[i:])/nsTrain
    
# # ================ Alternative Prediction ========================================== 
folderImages = './images/'
suptitle = 'Arch_2_Dataset_Seed_5{0}_zoom.{1}' 

plt.figure(1)
plt.title(suptitle.format('', ''))
for i in range(5):
    plt.plot(NlistModel, errors[1,:,i,0], '-o', label = 'E2_DNN dataset{0}'.format(i+1))
plt.plot(np.arange(160),errorPOD,'--', label = 'E2_POD')
plt.ylim(0.9e-6,0.35e-5)
plt.yscale('log')
plt.xlabel('N')
plt.ylabel('weighted mean square error')
plt.grid()
plt.legend(loc = 'best')
plt.savefig(folderImages + suptitle.format('_Error','png'))


# suptitle = 'Arch_1_Dataset_Seed_5_maxmin{0}_zoom.{1}' 

# plt.figure(2,(8,5))
# plt.title(suptitle.format('', ''))
# for i in range(5):
#     plt.plot(NlistModel, errors[0,:,i,2], '-o', label = 'E2_DNN(max) dataset{0}'.format(i+1))
#     plt.plot(NlistModel, errors[0,:,i,3], '-x', label = 'E2_DNN(min) dataset{0}'.format(i+1))
# plt.plot(np.arange(160),errorPOD,'--', label = 'E2_POD')
# plt.ylim(1.0e-10,0.1)
# plt.xlim(0,200)
# plt.yscale('log')
# plt.xlabel('N')
# plt.ylabel('weighted mean square error')
# plt.grid()
# plt.legend(loc = 'best')
# plt.savefig(folderImages + suptitle.format('_Error','png'))


# arch = 2
# Nrb = 140

# folderImages = './images/'
# suptitle = 'Historic_Arch_{0}_N{1}_Dataset_Seed_5'.format(arch,Nrb) 


# historic_file = './models/newArchitectures_cluster/new5/weights_ny{0}_arch{1}_history.csv'.format(Nrb,arch)

# hist = np.loadtxt(historic_file,skiprows=1,delimiter= ',')

# plt.figure(1)
# plt.title(suptitle)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.plot(hist[:,0] , hist[:,2],label='Train Loss')
# plt.plot(hist[:,0] , hist[:,6],label = 'Val loss')
# plt.yscale('log')
# plt.legend()
# plt.grid()
# plt.legend()    
# plt.savefig(folderImages + suptitle + '.png')

