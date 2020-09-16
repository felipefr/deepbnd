import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0,'../')
sys.path.insert(0,'../../utils/')
import matplotlib.pyplot as plt
import numpy as np

import myTensorflow as mytf
from timeit import default_timer as timer

import h5py
import pickle
import Generator as gene
import myHDF5 
import dataManipMisc as dman 
from sklearn.preprocessing import MinMaxScaler
import miscPosprocessingTools as mpt

import json

def getTraining(ns_start, ns_end, nX, nY, Xdatafile, Ydatafile, scalerX = None, scalerY = None):
    X = np.zeros((ns_end - ns_start,nX))
    Y = np.zeros((ns_end - ns_start,nY))
    
    for i in range(ns_end - ns_start):
        j = i + ns_start
        X[i,:] = np.loadtxt(Xdatafile.format(j))[:nX,2]
    
    Y = np.loadtxt(Ydatafile)[ns_start:ns_end,:nY] 
   

    if(type(scalerX) == type(None)):
        scalerX = MinMaxScaler()
        scalerX.fit(X)
    
    if(type(scalerY) == type(None)):
        scalerY = MinMaxScaler()
        scalerY.fit(Y)
            
    return scalerX.transform(X), scalerY.transform(Y), scalerX, scalerY


def basicModelTraining(nsTrain, nX, nY, net, fnames):
    X, Y, scalerX, scalerY = getTraining(0,nsTrain, nX, nY, fnames['prefix_in_X'], fnames['prefix_in_Y'])
    
    Neurons = net['Neurons']
    actLabel = net['activations']
    drps = net['drps']
    reg = net['reg']
    Epochs = net['epochs']
    decay = net['decay']
    lr = net['lr']
    
    model = mytf.DNNmodel(nX, nY, Neurons, actLabel = actLabel , drps = drps, lambReg = reg  )
    
    num_parameters = 0 # in case of treating differently a part of the outputs
    history = mytf.my_train_model( model, X, Y, num_parameters, Epochs, lr = lr, decay = decay, w_l = 1.0, w_mu = 0.0)
        
    mytf.plot_history( history, savefile = fnames['prefix_out'] + 'plot_history' + fnames['suffix_out'])
    
    # with open(partialRadical + 'history_twoPoints_dispBound' + Run_id + '.dat', 'wb') as f:
    #     pickle.dump(history.history, f)
        
    model.save_weights(savefile = fnames['prefix_out'] + 'weights' + fnames['suffix_out'])



# simul_id = int(sys.argv[1])
# EpsDirection = int(sys.argv[2])

simul_id = 3
EpsDirection = 1
base_offline_folder = "/Users/felipefr/EPFL/newDLPDES/DATA/deepBoundary/data{0}/".format(simul_id)

run_id = 1
case_id = 6

nX = 4 # because the only first 4 are relevant, the other are constant
nY = 5 # 10 alphas

print("starting with", simul_id, EpsDirection, case_id, run_id)
nsTrain = 1000

net = {'Neurons': 5*[128], 'drps': 7*[0.0], 'activations': ['relu','relu','relu'], 
       'reg': 0.00001, 'lr': 0.01, 'decay' : 0.5, 'epochs': 500}

       
fnames = {}      
fnames['suffix_out'] = '_{0}_{1}_{2}_{3}'.format(EpsDirection,nY,case_id,run_id)
fnames['prefix_out'] = 'saves_PODMSE_comparison/'
fnames['prefix_in_X'] = base_offline_folder + "ellipseData_{0}.txt"
fnames['prefix_in_Y'] = "Y_reference_{0}_{1}.txt".format(simul_id, EpsDirection)

os.system("mkdir " + fnames['prefix_out']) # in case the folder is not present



basicModelTraining(nsTrain, nX, nY, net, fnames)


# 1)
# Neurons= 4*[32]
# drps = 6*[0.0]
# lr2 = 0.00
# lr = 0.01
# decay = 0.5



# 2)
# Neurons= 4*[64]
# drps = 6*[0.0]
# lr2 = 0.000001
# lr = 0.01
# decay = 0.5



# 3)
# Neurons= 4*[64]
# drps = 6*[0.0]
# lr2 = 0.00001
# lr = 0.01
# decay = 0.5
# actLabel = ['relu','relu','linear']



# # 4)
# Neurons= 4*[64]
# drps = 6*[0.0]
# lr2 = 0.0001
# lr = 0.01
# decay = 1.0

# 5)
# Neurons= 5*[128]
# drps = 7*[0.0]
# lr2 = 0.0001
# lr = 0.01
# decay = 1.0



# 6)
# Neurons= 5*[128]
# drps = 7*[0.0]
# lr2 = 0.00001
# lr = 0.01
# decay = 0.5

# 7)
# Neurons= 6*[256]
# drps = 8*[0.0]
# lr2 = 0.00001
# lr = 0.01
# decay = 0.5

# 8)
# Neurons= 6*[512]
# drps = 8*[0.2]
# lr2 = 0.0
# lr = 0.001
# decay = 0.5

# 9)
# Neurons= 6*[512]
# drps = 8*[0.2]
# lr2 = 1.0e-4
# lr = 0.001
# decay = 0.5
# actLabel = ['relu','relu','sigmoid']

# 10)
# Neurons= 6*[512]
# drps = 8*[0.2]
# lr2 = 1.0e-7
# lr = 0.001
# decay = 0.5
# actLabel = ['leaky_relu','leaky_relu','leaky_relu']

# 11)
# Neurons= 6*[512]
# drps = 8*[0.2]
# lr2 = 1.0e-7
# lr = 0.01
# decay = 0.5
# actLabel = ['leaky_relu','leaky_relu','leaky_relu']


# 12)
# Neurons= 6*[512]
# drps = 8*[0.15]
# lr2 = 1.0e-7
# lr = 0.01
# decay = 0.7
# actLabel = ['leaky_relu','leaky_relu','leaky_relu']

