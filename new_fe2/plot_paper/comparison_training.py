import os, sys
sys.path.insert(0,'../../utils/')

# import generation_deepBoundary_lib as gdb
import matplotlib.pyplot as plt
import numpy as np

# import myTensorflow as mytf
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
# import tensorflow as tf

import plotUtils

# Test Loading 

rootDataPath = open('../../../rootDataPath.txt','r').readline()[:-1]

loadType = 'S'

folderTrain = rootDataPath + '/new_fe2/training/models/'
folderBasis = rootDataPath + '/new_fe2/dataset/'

NlistModel = [5,10,20,40,80,140]

arch = 2
Nrb = 140

historic_file = rootDataPath + '/new_fe2/training/models/weights_big_{0}_{1}_history.csv'.format(loadType,Nrb)

hist = np.loadtxt(historic_file,skiprows=1,delimiter= ',')


# # ======================= Reading theoretical POD =================== 
# nsTrain = 51200
# nbasis = 160
# nameEigen = folderBasis + 'Wbasis.h5'

# eig = myhd.loadhd5(nameEigen, 'sig_%s'%loadType)   
# errorPOD = np.zeros(nbasis)


# suptitle = 'Historic_Arch_{0}_N{1}_Dataset_Seed_5'.format(arch,Nrb) 

alpha = np.mean(hist[4990:,6])/np.mean(hist[2980:3020,6])
beta = -np.log(alpha)/2000
fac = alpha*np.ones(5000)
fac[3000:] = np.array([ alpha*np.exp(beta*i) for i in range(2000)])

plt.figure(1,(5.0,3.5))
plt.title("Loss Minimisation - Shear Model $\mathcal{N}^{(3)}$ - $N'_{rb} = %d$"%Nrb)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(hist[:,0] , hist[:,2],label='Training Loss')
plt.plot(hist[:,0] , fac*hist[:,6],label = 'Validation Loss')
plt.yscale('log')
plt.legend()
plt.grid()
plt.legend()    
plt.savefig('trainingShear_ny{0}.pdf'.format(Nrb))
plt.savefig('trainingShear_ny{0}.eps'.format(Nrb))

