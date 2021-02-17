nets = {}
# series with 5000 epochs
nets['35'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 5000, 'weightTsh' : 5} # normally reg = 1e-5

nets['36'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 5000, 'weightTsh' : 10} # normally reg = 1e-5

nets['37'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 5000, 'weightTsh' : 15} # normally reg = 1e-5

nets['38'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 5000, 'weightTsh' : 20} # normally reg = 1e-5

nets['39'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 5000, 'weightTsh' : 25} # normally reg = 1e-5

nets['40'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 5000, 'weightTsh' : 30} # normally reg = 1e-5

nets['41'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 5000, 'weightTsh' : 35} # normally reg = 1e-5

nets['42'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 5000, 'weightTsh' : 40} # normally reg = 1e-5

nets['43'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 5000, 'weightTsh' : 60} # normally reg = 1e-5

nets['44'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 5000, 'weightTsh' : 80} # normally reg = 1e-5

nets['45'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 5000, 'weightTsh' : 100} # normally reg = 1e-5

nets['46'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 5000, 'weightTsh' : 120} # normally reg = 1e-5

nets['47'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 5000, 'weightTsh' : 140} # normally reg = 1e-5

nets['48'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 5000, 'weightTsh' : 160} # normally reg = 1e-5

import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0,'../')
sys.path.insert(0,'../../utils/')

import generation_deepBoundary_lib as gdb
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
import meshUtils as meut

import json
import copy

import matplotlib.pyplot as plt
import symmetryLib as syml

f = open("../../../rootDataPath.txt")
rootData = f.read()[:-1]
f.close()

# Test Loading 

ntest  = 5120
folderTest = rootData + '/deepBoundary/testStress/P2/'
# folderTest = rootData + "/deepBoundary/smartGeneration/LHS_frozen_p4_volFraction/"
# folderTest = rootData + "/deepBoundary/smartGeneration/LHS_p4_fullSymmetric/"
nameYtest = folderTest + 'Y_all.h5'
# nameYtest = folderTest + 'Y.h5'
# nameYtest = folderTest + 'Y_svd_full.h5'

nameEllipseDataTest = folderTest + 'ellipseData_17.h5'
# nameEllipseDataTest = folderTest + 'ellipseData_1.h5'
# nameEllipseDataTest = folderTest + 'ellipseData_fullSymmetric.h5'


Ytest = myhd.loadhd5(nameYtest, 'Ylist')
ellipseDataTest = myhd.loadhd5(nameEllipseDataTest, 'ellipseData')


# Model Prediction
nX = 36
nY = 160
ns = 4*10240

# folderModels = rootData + '/deepBoundary/smartGeneration/newTrainingSymmetry/'
# folderTrain = rootData + "/deepBoundary/smartGeneration/LHS_frozen_p4_volFraction/"
# nameEllipseDataTrain = folderTrain + "ellipseData_1.h5"
# nameYtrain = folderTrain + "Y.h5"
# namenet = 'weights_LHS_p4_volFraction_drop02_nX36_nY140_{0}'

folderModels = rootData + '/deepBoundary/smartGeneration/newTrainingSymmetry/fullSymmetric/'
folderTrain = rootData + "/deepBoundary/smartGeneration/LHS_p4_fullSymmetric/"
nameEllipseDataTrain = folderTrain + "ellipseData_fullSymmetric.h5"
nameYtrain = folderTrain + "Y_svd_full.h5"

X, Y, scalerX, scalerY = syml.getTraining(0,ns, nX, nY, nameEllipseDataTrain, nameYtrain )
namenet = 'weightsfullSymmetric_save_0_nY{0}'


# models = []
# for j in range(35,49):
#     net =  nets[str(j)]        
#     models.append(mytf.DNNmodel(nX, nY, net['Neurons'], actLabel = net['activations'], drps = net['drps'], lambReg = net['reg']))
#     models[-1].load_weights( folderModels + namenet.format(j))
    
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


# maxAlpha = scalerY.data_max_
# minAlpha = scalerY.data_min_

# w_l = (maxAlpha - minAlpha)**2.0

# Nlist = [5,10,15,20,25,30,35,40,60,80,100,120,140]
# lossTest_snap = np.zeros((len(Nlist),ntest))
# lossTest_snap_rest = np.zeros((len(Nlist),ntest))

# for i, N in enumerate(Nlist):
#     for j in range(ntest):
#         for k in range(N):
#             lossTest_snap[i,j] = lossTest_snap[i,j] + w_l[k]*(Y_p_scaled[i][j,k] - Ytest_scaled[j,k])**2.0
        
#         for k in range(N):
#             lossTest_snap_rest[i,j] = lossTest_snap_rest[i,j] + w_l[140-k]*Ytest_scaled[j,140-k]**2.0

# lossTest = np.mean(lossTest_snap, axis = 1)
# lossTest_rest = np.mean(lossTest_snap_rest, axis = 1)





folder = rootData + '/deepBoundary/smartGeneration/newTrainingSymmetry/'
folder2 = rootData + '/deepBoundary/smartGeneration/LHS_frozen_p4_volFraction/'
folder3 = rootData + '/deepBoundary/smartGeneration/LHS_p4_fullSymmetric/'
nameMeshRefBnd = 'boundaryMesh.xdmf'
nameWbasis = folder2 + 'Wbasis_new.h5'
nameSnaps = folder2 + 'snapshots_all.h5'
nameY = folder2 + 'Y.h5'
nameout = 'plot_history_LHS_p4_volFraction_drop02_nX{0}_nY{1}_{2}_history_.txt'
nameout_val = 'plot_history_LHS_p4_volFraction_drop02_nX{0}_nY{1}_{2}_history_val.txt'

eig = myhd.loadhd5(folder2 + 'eigens.hd5', 'eigenvalues')   
errorPOD = np.zeros(160)



# eig_full_symmetric = myhd.loadhd5(folder3 + 'eigenvalues.hd5', 'eigenvalues')  
eig_full_symmetric = myhd.loadhd5(folder3 + 'eigens_svd_full_onlyEigenvalues.hd5', 'eigenvalues')  
errorPOD_full_symmetric = np.zeros(160)


# Mref = meut.EnrichedMesh(nameMeshRefBnd)
# Vref = VectorFunctionSpace(Mref,"CG", 1)
# dsRef = Measure('ds', Mref) 
# dotProduct = lambda u,v, dx : assemble(inner(u,v)*ds)

# Nlist = np.arange(2,160,8)


# Wbasis, fw = myhd.loadhd5_openFile(nameWbasis, 'Wbasis')
# Isol = myhd.loadhd5(nameSnaps,'solutions_trans')
# Ylist = myhd.loadhd5(nameY,'Ylist')

# errorPOD_brute = np.zeros((len(Nlist), ns)) 

# for i, N in enumerate(Nlist):
#     print("computing errors for N = ", N)
#     errorPOD_brute[i,:] = gdb.getErrors(N,Ylist,Wbasis, Isol, Vref, dsRef, dotProduct)


for i in range(160):
    errorPOD[i] = np.sum(eig[i:])/ns
    errorPOD_full_symmetric[i] = np.sum(eig_full_symmetric[i:])/(4*ns)
    


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

# for i in [15,16,17,18]:    
#     hist_1.append(np.loadtxt(folder + nameout.format(nX,nY,i) ))
#     hist_val_1.append(np.loadtxt(folder + nameout_val.format(nX,nY,i) ))

# for i in [19,20,21,22,23]:    
#     hist_2.append(np.loadtxt(folder + nameout.format(nX,nY,i) ))
#     hist_val_2.append(np.loadtxt(folder + nameout_val.format(nX,nY,i) ))

# for i in [26,24]:    
#     hist_3.append(np.loadtxt(folder + nameout.format(nX,nY,i) ))
#     hist_val_3.append(np.loadtxt(folder + nameout_val.format(nX,nY,i) ))

# for i in [27,25]:    
#     hist_4.append(np.loadtxt(folder + nameout.format(nX,nY,i) ))
#     hist_val_4.append(np.loadtxt(folder + nameout_val.format(nX,nY,i) ))

# for i in [29,28]:    
#     hist_3.append(np.loadtxt(folder + nameout.format(nX,nY,i) ))
#     hist_val_3.append(np.loadtxt(folder + nameout_val.format(nX,nY,i) ))

# for i in [8,9,10,11,12,13,14,1]:
#     hist_4.append(np.loadtxt(folder + nameout.format(nX,nY,i) ))
#     hist_val_4.append(np.loadtxt(folder + nameout_val.format(nX,nY,i) ))
    
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

# ====  new losses
folder_fullSymmetric = rootData + '/deepBoundary/smartGeneration/newTrainingSymmetry/fullSymmetric/logs_losses/'

nYlist_fullSymmetric = [5,10,15,20,25,30,35,40,60,80,100,120,140,160]

history_fullSymmetric = []

for ny in nYlist_fullSymmetric:
    history_fullSymmetric.append(np.loadtxt(folder_fullSymmetric + 'loss_log_ny{0}.txt'.format(ny)))

lastError_fullSymmetric = [history_fullSymmetric[i][-1,0] for i in range(len(nYlist_fullSymmetric))]
lastError_fullSymmetric_val = [history_fullSymmetric[i][-1,1] for i in range(len(nYlist_fullSymmetric))]

plt.figure(1)
# plt.plot(Nlist_1_2[:-1],lastError_1, '-o', label = 'ErrorDNN_train 1')
# plt.plot(Nlist_1_2,lastError_2, '-o', label = 'ErrorDNN_train 2')
# plt.plot(Nlist_3,lastError_3, '-o', label = 'ErrorDNN_train 3')
# plt.plot(Nlist_4,lastError_4, '-o', label = 'ErrorDNN_train 4')
# plt.plot(Nlist_5,lastError_5, '-o', label = 'ErrorDNN_train')

# plt.plot(Nlist_1_2[:-1],lastError_val_1, '-o', label = 'ErrorDNN_val 1')
# plt.plot(Nlist_1_2,lastError_val_2, '-o', label = 'ErrorDNN_val 2')
# plt.plot(Nlist_3,lastError_val_3, '-o', label = 'ErrorDNN_val 3')
# plt.plot(Nlist_4,lastError_val_4, '-o', label = 'ErrorDNN_val 4')
# plt.plot(Nlist_5,lastError_val_5, '-o', label = 'ErrorDNN_val 5')

plt.plot(Nlist_5,lastError_5, '-o', label = 'ErrorDNN')
plt.plot(np.arange(160),errorPOD,'--', label = 'ErrorPOD')
plt.plot(np.arange(160),errorPOD_full_symmetric,'--', label = 'ErrorPOD FS')
plt.plot(nYlist_fullSymmetric,lastError_fullSymmetric,'-o', label = 'ErrorDNN FS')
# plt.plot(np.arange(160),np.sqrt(errorPOD_full_symmetric/errorPOD_full_symmetric[0]),'--', label = 'ErrorPOD FS')
# plt.plot(nYlist_fullSymmetric,np.sqrt(lastError_fullSymmetric/errorPOD_full_symmetric[0]),'-o', label = 'ErrorDNN FS')
# plt.plot(nYlist_fullSymmetric,lastError_fullSymmetric_val,'-', label = 'ErrorDNN_val FS')
# plt.plot(Nlist_5,errorPOD[Nlist_5] + lastError_5, '-o', label = 'Total Error')
# plt.plot(nYlist_fullSymmetric[:-1], errorPOD_full_symmetric[nYlist_fullSymmetric[:-1]] + lastError_fullSymmetric[-1], '-o', label = 'Total Error FS')
# plt.plot(nYlist_fullSymmetric[:-1], np.sqrt((errorPOD_full_symmetric[nYlist_fullSymmetric[:-1]] + lastError_fullSymmetric[-1])/errorPOD_full_symmetric[0]), '-o', label = 'Total Error FS')
plt.plot(Nlist, lossTest_mse, '-o', label = 'Error Test (vs. NN FS)')
plt.ylim(1.0e-10,0.1)
# plt.xlim(20,80)
plt.yscale('log')
plt.xlabel('N')
plt.ylabel('weighted mean square error')
plt.grid()
plt.legend()
# plt.savefig('comparison_fullSymmetry_loss_POD_zoom.png')
# plt.savefig('fullSymmetry_loss_POD_total_relative.png')

plt.figure(2)
plt.plot(errorPOD)
plt.yscale('log')

