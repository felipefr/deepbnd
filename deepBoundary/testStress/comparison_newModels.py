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
import generation_deepBoundary_lib as gdb

import json
import copy

import matplotlib.pyplot as plt
import symmetryLib as syml
import tensorflow as tf

# Test Loading 


hybridOrExtended = 1

# folderTest = './models/dataset_newTest2/'
# folderTest = './models/dataset_newTest3/'
folderTest = './models/dataset_test/'
# folderTest = './models/dataset_extendedSymmetry_recompute/'
# folderTest = './models/dataset_hybrid/'

folderTrain = ['./models/dataset_hybrid/','./models/dataset_extendedSymmetry_recompute/'][hybridOrExtended]

idModel = 1
folderModels = './models/newModels/' + ['extended/','extended_bug/', 'half/', 'hybrid_linear/', 
                                        'hybrid_sigmoid/', 'simple_testAsValidation/'][idModel]

namenet = 'weights_ny{0}.hdf5'
NlistModel = [5,10,20,40,80]

nameYtest = folderTest + 'Y_{0}.h5'.format(['hybrid','extended'][hybridOrExtended])
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

nbasis = Ytest.shape[1]
print(nbasis)

nX = 36
nYmax = 160
X, Y, scalerX, scalerY = syml.getDatasets(nX, nYmax, nameEllipseDataTrain, nameYtrain )
nsTrain = len(Y)

# ======================= Reading Prepared Losses (Validate with brute force computed error) =================== 

# ns = [10240,4*10240][simpleOrExtended]
# nameHist = './models/extendedSymmetry/histories/loss_log_ny{0}.txt'

# NlistModel = [5,20,40,140]
# lastError = np.array([np.loadtxt(nameHist.format(ny,''))[-1] for ny in NlistModel])
# lastError_val = np.array([np.loadtxt(nameHist.format(ny,'val'))[-1] for ny in NlistModel])

eig = myhd.loadhd5(nameEigenTrain, 'eigenvalues')   
errorPOD = np.zeros(nbasis)

for i in range(nbasis):
    errorPOD[i] = np.sum(eig[i:])/nsTrain
    


# # ============== Error Brute Force ================================================

# nameMeshRefBnd = 'boundaryMesh.xdmf'
# Mref = meut.EnrichedMesh(nameMeshRefBnd)
# Vref = VectorFunctionSpace(Mref,"CG", 1)
# dsRef = Measure('ds', Mref) 
# dotProduct = lambda u,v, dx : assemble(inner(u,v)*ds)

Wbasis_M = myhd.loadhd5(nameWbasisTrain, ['Wbasis','massMatrix']) # for the simple case this comes in blocks 
# Isol = myhd.loadhd5(nameSnapsTrain,'solutions_trans') 
# Ylist = myhd.loadhd5(nameYtrain,'Ylist') # for the simple case this comes in blocks 

# # r = gdb.testOrthonormalityBasis(10, Wbasis_M[0], Vref, dsRef, dotProduct, Nmax = 50)
# # errorSymmetry = gdb.testSymmetryBasis(10, Wbasis_M, Vref)

# errorL2_mse_POD = gdb.getMSE_fast(Nlist,Ytest, Wbasis_M , IsolTest) 

# ================ Alternative Prediction ========================================== 


netSig = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 'reg': 0.0}  # normally reg = 1e-5
netLin = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['tanh','relu','linear'], 'reg': 0.0}  # normally reg = 1e-5

net = [netSig,netLin][0]

models = []
for nYi in NlistModel:      
    models.append(mytf.DNNmodel_notFancy(nX, nYi, net['Neurons'], net['activations']))
    models[-1].load_weights( folderModels + namenet.format(nYi))
    
# Prediction 
Xtest_scaled = scalerX.transform(ellipseDataTest[:,:,2])
Ytest_scaled = scalerY.transform(Ytest[:,:])
Y_p_scaled = []
Y_p = []

for i in range(len(models)):
    print('predictig model ', i )
    Y_p_scaled.append(models[i].predict(Xtest_scaled))
    Y_p.append(scalerY.inverse_transform(np.concatenate((Y_p_scaled[-1] , np.zeros((len(Xtest_scaled), nYmax - NlistModel[i]))), axis = 1)))


# ================ Computing the error ========================================== 
lossTest_mse = []
maxAlpha = scalerY.data_max_
minAlpha = scalerY.data_min_
w_l = (maxAlpha - minAlpha)**2.0  
for i, N in enumerate(NlistModel):
    lossTest_mse.append( mytf.custom_loss_mse_2(Y_p_scaled[i][:,:N], Ytest_scaled[:,:N], weight = w_l[:N]).numpy())
    

# brute force
errorL2_mse_total = []
for i, N in enumerate(NlistModel):
    errorL2_mse_total.append(gdb.getMSE_fast([N],Y_p[i][:,:N], Wbasis_M , IsolTest)[0]) 


suptitle = 'Model DNN: extendedSymmetric (LHS 10240), Test: LHS 5120 (seed=17)'
savefigure = '{0}_DNNextended_TestNew1.png'


plt.figure(1)
plt.title(suptitle)
plt.plot(NlistModel, errorL2_mse_total, '-o', label = 'ErrorDNN + ErrorPOD (brute force)')
plt.plot(NlistModel,lossTest_mse, '-x', label = 'ErrorDNN')
plt.plot(np.arange(160),errorPOD,'--', label = 'ErrorPOD')
plt.ylim(1.0e-10,0.1)
plt.yscale('log')
plt.xlabel('N')
plt.ylabel('weighted mean square error')
plt.grid()
plt.legend(loc = 'best')
plt.savefig(savefigure.format('Error'))


# Histogram

plt.figure(2,(18,10))
plt.suptitle(suptitle)
plt.subplot('321')
plt.title('Histogram Y_1 (test)')
plt.hist(Ytest[:,0],bins = 20)

plt.subplot('322')
plt.title('Histogram Y_1 (prediction)')
plt.hist(Y_p[2][:,0], bins = 20)

plt.subplot('323')
plt.title('Histogram Y_2 (test)')
plt.hist(Ytest[:,1],bins = 20)  

plt.subplot('324')
plt.title('Histogram Y_2 (prediction)')
plt.hist(Y_p[2][:,1], bins = 20)


plt.subplot('325')
plt.title('Histogram Y_3 (test)')
plt.hist(Ytest[:,2],bins = 20)

plt.subplot('326')
plt.title('Histogram Y_3 (prediction)')
plt.hist(Y_p[2][:,2], bins = 20)

plt.savefig(savefigure.format('histograms_1-3'))

plt.figure(3,(18,10))
plt.suptitle(suptitle)
plt.subplot('321')
plt.title('Histogram Y_4 (test = train)')
plt.hist(Ytest[:,3],bins = 20)

plt.subplot('322')
plt.title('Histogram Y_4 (prediction)')
plt.hist(Y_p[2][:,3], bins = 20)

plt.subplot('323')
plt.title('Histogram Y_5 (test = train)')
plt.hist(Ytest[:,4],bins = 20)

plt.subplot('324')
plt.title('Histogram Y_5 (prediction)')
plt.hist(Y_p[2][:,4], bins = 20)


plt.subplot('325')
plt.title('Histogram Y_6 (test = train)')
plt.hist(Ytest[:,5],bins = 20)

plt.subplot('326')
plt.title('Histogram Y_6 (prediction)'  )
plt.hist(Y_p[2][:,5], bins = 20)

plt.savefig(savefigure.format('histograms_4-6'))


