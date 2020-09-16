import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0,'../')
sys.path.insert(0,'../../utils/')
import matplotlib.pyplot as plt
import numpy as np

import myTensorflow as mytf
import tensorflow as tf
from timeit import default_timer as timer

import h5py
import pickle
import Generator as gene
import myHDF5 
import dataManipMisc as dman 
from sklearn.preprocessing import MinMaxScaler
import miscPosprocessingTools as mpt

import json

from basicTraining_lib import *

tf.config.optimizer.set_jit(True)

# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)



# simul_id = int(sys.argv[1])
# EpsDirection = int(sys.argv[2])

simul_id = 3
EpsDirection = 0
base_offline_folder = "/Users/felipefr/EPFL/newDLPDES/DATA/deepBoundary/data{0}/".format(simul_id)

arch_id = 1

nX = 4 # because the only first 4 are relevant, the other are constant

print("starting with", simul_id, EpsDirection, arch_id)
nsTrain = 1000

# Arch 1
net = {'Neurons': 5*[128], 'drps': 7*[0.0], 'activations': ['relu','relu','relu'], 
        'reg': 0.00001, 'lr': 0.01, 'decay' : 0.5, 'epochs': 1000}

# Arch 2
# net = {'Neurons': [128,256,256,256,512,1024], 'drps': 8*[0.1], 'activations': ['relu','relu','relu'], 
#        'reg': 0.0001, 'lr': 0.001, 'decay' : 0.5, 'epochs': 1000}

       
fnames = {}      

fnames['prefix_out'] = 'saves_correct_PODMSE_L2bnd_Eps{0}_arch{1}/'.format(EpsDirection, arch_id)
fnames['prefix_in_X'] = base_offline_folder + "ellipseData_{0}.txt"
fnames['prefix_in_Y'] = "./definitiveBasis/Y_L2bnd_original_{0}_{1}.hd5".format(simul_id, EpsDirection)

os.system("mkdir " + fnames['prefix_out']) # in case the folder is not present


# nYlist = np.array([2,5,10,20,50,80,110,150]).astype('int')
nYlist = np.array([2,5,8,12,16,20,25,30,35,40]).astype('int')
# nYlist = np.array([35,40]).astype('int')
Nruns = 2

for i, nY in enumerate(nYlist[0::3]):
    for j in range(1,Nruns):
        fnames['suffix_out'] = '_{0}_{1}'.format(nY,j)
        np.random.seed(j+1)
        tf.random.set_random_seed(i)
        mytf.dfInitK , mytf.dfInitB = mytf.set_seed_default_initialisers(j)
        hist, model = basicModelTraining(nsTrain, nX, nY, net, fnames)


