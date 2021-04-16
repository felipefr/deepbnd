import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0,'../utils/')
import matplotlib.pyplot as plt
import numpy as np

import myTensorflow as mytf
from timeit import default_timer as timer

from dolfin import *
import h5py
import pickle
# import Generator as gene
import myHDF5 
import dataManipMisc as dman 
from sklearn.preprocessing import MinMaxScaler
import miscPosprocessingTools as mpt
import myHDF5 as myhd
# import symmetryLib as syml
import tensorflow as tf
import multiphenicsMultiscale as mpms
import elasticity_utils as elut

from tensorflow_for_training import *
import meshUtils as meut
import fenicsMultiscale as fmts
import fenicsUtils as feut

# Nrb = int(sys.argv[1])
# Nrb = int(input("Nrb="))

typeModel = 'shear'
archLabel = '1'
archId = 1
nX = 36
nY = 3

folderBasis = './models/dataset_{0}1/'.format(typeModel)
folderTest = './models/dataset_{0}2/'.format(typeModel)
nameSnaps = folderTest + 'snapshots.h5'
nameXYtest = folderTest +  'XY_stress.h5'
nameScaleXY = folderBasis +  'scaler_stress.txt'

scalerX, scalerY  = importScale(nameScaleXY, nX, nY)

X = myhd.loadhd5(nameXYtest, 'X')
X_s = scalerX.transform(X)

nets = {}
nets[3] = {'Neurons': 2*[50], 'activations': 2*['swish'] + ['linear'], 'lr': 5.0e-3, 'decay' : 0.1, 'drps' : [0.0] + 2*[0.005] + [0.0], 'reg' : 1.0e-8}
nets[2] = {'Neurons': [40], 'activations': 1*['swish'] + ['linear'], 'lr': 5.0e-3, 'decay' : 0.1, 'drps' : 3*[0.0], 'reg' : 0.0}
nets[1] = {'Neurons': 3*[300], 'activations': 3*['swish'] + ['linear'], 'lr': 5.0e-3, 'decay' : 0.1, 'drps' : [0.0] + 3*[0.005] + [0.0], 'reg' : 1.0e-8}

net = nets[archId]

net['nY'] = nY
net['nX'] = nX
net['file_weights'] = folderBasis + 'models/stress/weights_arch{0}.hdf5'.format(archLabel)


# Prediction 
model = generalModel_dropReg(nX, nY, net)   

model.load_weights(net['file_weights'])

Y_p_s = model.predict(X_s)
Y_p = scalerY.inverse_transform(Y_p_s)

ns = len(Y_p)
ns_max = ns
np.random.seed(1)
randInd = np.arange(ns,dtype='int') 
# np.random.shuffle(randInd)
randInd =  randInd[:ns_max]

os.system('rm sigma_prediction_{0}_stress_direct_arch{1}_val.hd5'.format(typeModel,archLabel))
fields, f = myhd.zeros_openFile('sigma_prediction_{0}_stress_direct_arch{1}_val.hd5'.format(typeModel,archLabel),
                                [(ns_max,3),(ns_max,3),(ns_max,1)], ['sigma','sigma_ref','error'])
sigma, sigma_ref, error = fields  


# sigma_ref[:,:] = myhd.loadhd5('sigma_prediction_ny5_{0}.hd5'.format(typeModel), 'sigma_ref')[:,:]
sigma_ref[:,:] = myhd.loadhd5(nameSnaps, 'sigma')[randInd,:]


normStress = lambda s : np.sqrt(s[0]**2 + s[1]**2 + 2*s[2]**2)

for i, ii in enumerate(randInd):  
    print("snaptshots i, ii = ", i, ii)

    sigma[i,:] = Y_p[i,:]
    error[i,0] = normStress(sigma[i,:] - sigma_ref[i,:])
    
    print(sigma[i,:], sigma_ref[i,:],  error[i,0])

print(np.mean(error))
print(np.mean(error[:,0]**2))
f.close()