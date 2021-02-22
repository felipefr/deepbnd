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
# from dolfin import *
# import generation_deepBoundary_lib as gdb

import json
import copy

import matplotlib.pyplot as plt
import symmetryLib as syml
import tensorflow as tf

# Test Loading 

simpleOrExtended = 1

nbasis = 160
ntest  = 5120
# folderTest = './models/dataset_test/'
folderTest = './models/dataset_extendedSymmetry_recompute/'
folderTrain = ['./models/dataset_simple/','./models/dataset_extendedSymmetry_recompute/'][simpleOrExtended]

nameYtest = folderTest + 'Y.h5'
nameSnapsTest = folderTest + 'snapshots.h5'
nameYtrain = folderTrain + 'Y.h5'
nameWbasisTrain = folderTrain + 'Wbasis.h5'
nameSnapsTrain = folderTrain + 'snapshots.h5'
nameEigenTrain = folderTrain + 'eigens.hd5'

nameEllipseDataTest = folderTest + 'ellipseData.h5'
nameEllipseDataTrain = folderTrain + 'ellipseData.h5'

Ytest = myhd.loadhd5(nameYtest, 'Ylist')
ellipseDataTest = myhd.loadhd5(nameEllipseDataTest, 'ellipseData')
IsolTest = myhd.loadhd5(nameSnapsTest, 'solutions_trans')

# ======================= Reading Prepared Losses (Validate with brute force computed error) =================== 

ns = [10240,4*10240][simpleOrExtended]
nameHist = ['./models/simple/histories/loss_log_ny{0}.txt' , './models/extendedSymmetry/histories/loss_log_ny{0}.txt'][simpleOrExtended]

# # for i in [35,36,37,38,39,40,41,42,31,32,33,34]: #redo with 31 - 34
Nlist = [[5,10,15,20,25,30,35,40,80,100,120,140], [5,10,15,20,25,30,35,40,60,80,100,120,140,160]][simpleOrExtended]

lastError = np.array([np.loadtxt(nameHist.format(ny))[-1,0] for ny in Nlist])
lastError_val = np.array([np.loadtxt(nameHist.format(ny))[-1,1] for ny in Nlist])

eig = myhd.loadhd5(nameEigenTrain, 'eigenvalues')   
errorPOD = np.zeros(nbasis)

for i in range(nbasis):
    errorPOD[i] = np.sum(eig[i:])/ns
    


# # ============== Error Brute Force ================================================

# nameMeshRefBnd = 'boundaryMesh.xdmf'
# Mref = meut.EnrichedMesh(nameMeshRefBnd)
# Vref = VectorFunctionSpace(Mref,"CG", 1)
# dsRef = Measure('ds', Mref) 
# dotProduct = lambda u,v, dx : assemble(inner(u,v)*ds)

# Wbasis_M = myhd.loadhd5(nameWbasisTrain, ['Wbasis','massMatrix']) # for the simple case this comes in blocks 
# Isol = myhd.loadhd5(nameSnapsTrain,'solutions_trans') 
# Ylist = myhd.loadhd5(nameYtrain,'Ylist') # for the simple case this comes in blocks 

# # r = gdb.testOrthonormalityBasis(10, Wbasis_M[0], Vref, dsRef, dotProduct, Nmax = 50)
# # errorSymmetry = gdb.testSymmetryBasis(10, Wbasis_M, Vref)

# errorL2_mse_POD = gdb.getMSE_fast(Nlist,Ytest, Wbasis_M , IsolTest) 

# ============================= Model Prediction ==========================================
# nX = 36
# nY = 160
# ns = 4*10240

# folderModels = ['./models/simple/ny140_arch35to48_epochs5000/','./models/extendedSymmetry/'][simpleOrExtended]
# namenet = 'weights_nY{0}'
# NlistModel = [[5,10,15,20,25,30,35,40,80,100,120,140], [5,20,40,140]][simpleOrExtended]


# X, Y, scalerX, scalerY = syml.getTraining(0,ns, nX, nY, nameEllipseDataTrain, nameYtrain )

# net = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 'reg': 0.0}  # normally reg = 1e-5

# models = []

# for j in NlistModel:  
#     models.append(mytf.DNNmodel(nX, j, net['Neurons'], actLabel = net['activations'], drps = net['drps'], lambReg = net['reg']))
#     models[-1].load_weights( folderModels + namenet.format(j))
    
# # # Prediction
# Xtest_scaled = scalerX.transform(ellipseDataTest[:,:,2])
# Ytest_scaled = scalerY.transform(Ytest[:,:])
# Y_p_scaled = []
# Y_p = []

# for i in range(len(models)):
#     print('predictig model ', i )
#     Y_p_scaled.append(models[i].predict(Xtest_scaled))
#     # Y_p.append(scalerY.inverse_transform(Y_p_scaled[-1]))
    

# maxAlpha = scalerY.data_max_
# minAlpha = scalerY.data_min_

# w_l = (maxAlpha - minAlpha)**2.0

# # Nlist = [5,20,40,140]
# lossTest_snap = np.zeros((len(Nlist),ntest))
# lossTest_snap_rest = np.zeros((len(Nlist),ntest))

# for i, N in enumerate(Nlist):
#     for j in range(ntest):
#         for k in range(N):
#             lossTest_snap[i,j] = lossTest_snap[i,j] + w_l[k]*(Y_p_scaled[i][j,k] - Ytest_scaled[j,k])**2.0
        
#         # for k in range(N):
#         #     lossTest_snap_rest[i,j] = lossTest_snap_rest[i,j] + w_l[140-k]*Ytest_scaled[j,140-k]**2.0



# lossTest = np.mean(lossTest_snap, axis = 1)
# # # lossTest_rest = np.mean(lossTest_snap_rest, axis = 1)

# ================ Alternative Prediction ========================================== 
nX = 36
nY = 160
ns = 4*10240

# folderModels = './models/extendedSymmetry/'
folderModels = './models/extendedSymmetry/'
NlistModel = [5,20,40,140]


X, Y, scalerX, scalerY = syml.getTraining(0,ns, nX, nY, nameEllipseDataTrain, nameYtrain )
namenet = 'weights_nY{0}'
    
net = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 'reg': 0.0}  # normally reg = 1e-5
# net = {'Neurons': 3*[50], 'activations': ['relu','relu','sigmoid'], 'lr': 1.0e-4, 'decay' : 1.0} # normally reg = 1e-5
models = []
for nYi in NlistModel:      
    models.append(mytf.DNNmodel(nX, nY, net['Neurons'], actLabel = net['activations'], drps = net['drps'], lambReg = net['reg']))
    # models.append(mytf.DNNmodel_notFancy(nX, nY, net['Neurons'], net['activations']))
    models[-1].load_weights( folderModels + namenet.format(nYi))
    
    # models[-1] = tf.keras.models.load_model(folderModels + namenet.format(nYi))

        
    
# Prediction 
Xtest_scaled = scalerX.transform(ellipseDataTest[:,:,2])
Ytest_scaled = scalerY.transform(Ytest[:,:])
Y_p_scaled = []
Y_p = []

for i in range(len(models)):
    print('predictig model ', i )
    Y_p_scaled.append(models[i].predict(Xtest_scaled))
    Y_p.append(scalerY.inverse_transform(Y_p_scaled[-1]))


lossTest_mse = []
maxAlpha = scalerY.data_max_
minAlpha = scalerY.data_min_
w_l = (maxAlpha - minAlpha)**2.0  
for i, N in enumerate(NlistModel):
    lossTest_mse.append( mytf.custom_loss_mse_2(Y_p_scaled[i][:,:N], Ytest_scaled[:,:N], weight = w_l[:N]).numpy())


plt.figure(1)

plt.plot(NlistModel,lossTest_mse, '-o', label = 'ErrorDNN')
plt.plot(np.arange(160),errorPOD,'--', label = 'ErrorPOD')
plt.ylim(1.0e-10,0.1)
plt.yscale('log')
plt.xlabel('N')
plt.ylabel('weighted mean square error')
plt.grid()
plt.legend()

# plt.figure(3)

# errorL2_mse_POD_T = [errorL2_mse_POD,errorL2_mse_POD_2,errorL2_mse_POD_3,errorL2_mse_POD_4]

# plt.title('POD extendend symmetric recomputed (test + mirroed)')
# for i in range(4):
#     plt.plot(Nlist,errorL2_mse_POD_T[i], '-o', label = 'ErrorPOD (brute force) mirroed T{0}'.format(i))
# plt.plot(np.arange(160),errorPOD,'--', label = 'ErrorPOD (eigenvalues)')
# plt.ylim(1.0e-10,0.1)
# plt.yscale('log')
# plt.xlabel('N')
# plt.ylabel('weighted mean square error')
# plt.grid()
# plt.legend()

# plt.figure(4)

# plt.title('POD extended symmetric recomputed (test)')
# plt.plot(Nlist,errorL2_mse_POD, '-o', label = 'ErrorPOD (brute force) - Test'.format(i))
# plt.plot(np.arange(160),errorPOD,'--', label = 'ErrorPOD (eigenvalues)')
# plt.ylim(1.0e-10,0.1)
# plt.yscale('log')
# plt.xlabel('N')
# plt.ylabel('weighted mean square error')
# plt.grid()
# plt.legend()