nets = {}
# series with 5000 epochs
nets['35'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 5000, 'weightTsh' : 5} # normally reg = 1e-5

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
# import meshUtils as meut

import json
import copy

import matplotlib.pyplot as plt
import symmetryLib as syml

f = open("../../../rootDataPath.txt")
rootData = f.read()[:-1]
f.close()

# Test Loading 

ntest  = 5120
folderTest = './models/dataset_test/'
nameYtest = folderTest + 'Y_extended.h5'

nameEllipseDataTest = folderTest + 'ellipseData.h5'

Ytest = myhd.loadhd5(nameYtest, 'Ylist')
ellipseDataTest = myhd.loadhd5(nameEllipseDataTest, 'ellipseData')


# Model Prediction
nX = 36
nY = 160
ns = 4*10240

folderModels = './models/extendedSymmetry/'
folderTrain = './models/dataset_extendedSymmetry_recompute/'
nameEllipseDataTrain = folderTrain + "ellipseData.h5"
nameYtrain = folderTrain + "Y.h5"

X, Y, scalerX, scalerY = syml.getTraining(0,ns, nX, nY, nameEllipseDataTrain, nameYtrain )
# namenet = 'weightsfullSymmetric_save_0_nY{0}'
namenet = 'weights_nY{0}'
    
models = []
for nYi in [5,20,40,140]:
    net =  nets['35']        
    models.append(mytf.DNNmodel(nX, nY, net['Neurons'], actLabel = net['activations'], drps = net['drps'], lambReg = net['reg']))
    models[-1].load_weights( folderModels + namenet.format(nYi))
        
    
# Prediction 
Xtest_scaled = scalerX.transform(ellipseDataTest[:,:,2])
Ytest_scaled = scalerY.transform(Ytest[:,:])
Y_p_scaled = []
Y_p = []

for i in range(len(models)):
    print('predictig model ', i )
    Y_p_scaled.append(models[i].predict(Xtest_scaled))
    Y_p.append(scalerY.inverse_transform(Y_p_scaled[-1]))


maxAlpha = scalerY.data_max_
minAlpha = scalerY.data_min_

w_l = (maxAlpha - minAlpha)**2.0

# Nlist = [5,10,15,20,25,30,35,40,60,80,100,120,140,160]
Nlist = [5,20,40,140]
lossTest_snap = []
lossTest_mse = []

for i, N in enumerate(Nlist):
    lossTest_snap.append( np.sum( w_l[:N]*(Y_p_scaled[i][:,:N] - Ytest_scaled[:,:N])**2, axis = 1 )  )
    lossTest_mse.append( np.mean(lossTest_snap[-1])  )


plt.figure(1)
plt.plot(Nlist, lossTest_mse, '-o', label = 'Error Test (vs. NN FS)')
plt.ylim(1.0e-10,0.1)
plt.yscale('log')
plt.xlabel('N')
plt.ylabel('weighted mean square error')
plt.grid()
plt.legend()


