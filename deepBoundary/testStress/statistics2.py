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
from dolfin import *

# Load mesh Boundary 
nameMeshRefBnd = 'boundaryMesh.xdmf'
Mref = meut.EnrichedMesh(nameMeshRefBnd)
Vref = VectorFunctionSpace(Mref,"CG", 1)
dsRef = Measure('ds', Mref) 
dotProduct = lambda u,v, dx : assemble(inner(u,v)*ds)

# Test Loading 
folderTest = './models/dataset_test/'
nameYtest = folderTest + 'Y_extended.h5'
nameEllipseDataTest = folderTest + 'ellipseData.h5'
nameSnapsTest = folderTest + 'snapshots.h5'

Ytest = myhd.loadhd5(nameYtest, 'Ylist')
ellipseDataTest = myhd.loadhd5(nameEllipseDataTest, 'ellipseData')
IsolT = myhd.loadhd5(nameSnapsTest,['solutions_trans'])[0]

# Train Loading
folderTrain = './models/dataset_extendedSymmetry_recompute/'
nameYtrain = folderTrain + "Y.h5"
nameEllipseDataTrain = folderTrain + "ellipseData.h5"
nameSnapsTrain = folderTrain + 'snapshots.h5'

Ytrain = myhd.loadhd5(nameYtrain, 'Ylist')
ellipseDataTrain = myhd.loadhd5(nameEllipseDataTest, 'ellipseData')
Isol = myhd.loadhd5(nameSnapsTrain,['solutions_trans'])[0]

# Plotting solutions
# u1 = Function(Vref)
# u2 = Function(Vref)

# u1.vector().set_local(IsolT[0,:])
# u2.vector().set_local(Isol[0,:])

# plt.figure(1)
# plot(Mref)

plt.figure(1,(18,10))
plt.subplot('321')
plt.title('Histogram Y_1 (test)')
plt.hist(Ytest[:,0],bins = 20)

plt.subplot('322')
plt.title('Histogram Y_1 (train 0:10240)')
plt.hist(Ytrain[:10240,0], bins = 20)

plt.subplot('323')
plt.title('Histogram Y_2 (test)')
plt.hist(Ytest[:,1],bins = 20)

plt.subplot('324')
plt.title('Histogram Y_2 (train 0:10240)')
plt.hist(Ytrain[:10240,1], bins = 20)


plt.subplot('325')
plt.title('Histogram Y_3 (test)')
plt.hist(Ytest[:,2],bins = 20)

plt.subplot('326')
plt.title('Histogram Y_3 (train 0:10240)')
plt.hist(Ytrain[:10240,2], bins = 20)

plt.savefig('histograms_train_vs_test-0-2_notSymmetrised.png')

plt.figure(2,(18,10))
plt.subplot('321')
plt.title('Histogram Y_4 (test)')
plt.hist(Ytest[:,3],bins = 20)

plt.subplot('322')
plt.title('Histogram Y_4 (train 0:10240)')
plt.hist(Ytrain[:10240,3], bins = 20)

plt.subplot('323')
plt.title('Histogram Y_5 (test)')
plt.hist(Ytest[:,4],bins = 20)

plt.subplot('324')
plt.title('Histogram Y_5 (train 0:10240)')
plt.hist(Ytrain[:10240,4], bins = 20)


plt.subplot('325')
plt.title('Histogram Y_6 (test)')
plt.hist(Ytest[:,5],bins = 20)

plt.subplot('326')
plt.title('Histogram Y_6 (train 0:10240)')
plt.hist(Ytrain[:10240,5], bins = 20)

plt.savefig('histograms_train_vs_test-3-5_notSymmetrised.png')


# # plt.plot(np.mean(Ytrain,axis=0))
# plt.plot(np.abs(np.mean(Ytrain,axis=0)), label = 'train')
# plt.plot(np.abs(np.mean(Ytrain,axis=0)) + np.std(Ytrain,axis=0), '--', label = 'train + std')
# plt.plot(np.abs(np.mean(Ytrain,axis=0)) - np.std(Ytrain,axis=0), '--', label = 'train + std')

# plt.yscale('log')
# plt.legend(loc= 'best')

# plt.show()