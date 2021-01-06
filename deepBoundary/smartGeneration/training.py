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
# import Generator as gene
import myHDF5 
import dataManipMisc as dman 
from sklearn.preprocessing import MinMaxScaler
import miscPosprocessingTools as mpt
import myHDF5 as myhd

import json

def getTraining(ns_start, ns_end, nX, nY, Xdatafile, Ydatafile, scalerX = None, scalerY = None):
    X = np.zeros((ns_end - ns_start,nX))
    Y = np.zeros((ns_end - ns_start,nY))
    
    X = myhd.loadhd5(Xdatafile, 'ellipseData')[ns_start:ns_end,:nX,2]
    Y = myhd.loadhd5(Ydatafile, 'Ylist')[ns_start:ns_end,:nY]
    
    if(type(scalerX) == type(None)):
        scalerX = MinMaxScaler()
        scalerX.fit(X)
    
    if(type(scalerY) == type(None)):
        scalerY = MinMaxScaler()
        scalerY.fit(Y)
            
    return scalerX.transform(X), scalerY.transform(Y), scalerX, scalerY


def basicModelTraining(nsTrain, nX, nY, net, fnames, w_l = 1.0):
    X, Y, scalerX, scalerY = getTraining(0,nsTrain, nX, nY, fnames['prefix_in_X'], fnames['prefix_in_Y'])
    
    Neurons = net['Neurons']
    actLabel = net['activations']
    drps = net['drps']
    reg = net['reg']
    Epochs = net['epochs']
    decay = net['decay']
    lr = net['lr']
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    X = X[indices,:]
    Y = Y[indices,:]
    
    model = mytf.DNNmodel(nX, nY, Neurons, actLabel = actLabel , drps = drps, lambReg = reg  )
    
    num_parameters = 0 # in case of treating differently a part of the outputs
    history = mytf.my_train_model( model, X, Y, num_parameters, Epochs, lr = lr, decay = decay, w_l = w_l, w_mu = 0.0)
        
    mytf.plot_history( history, savefile = fnames['prefix_out'] + 'plot_history' + fnames['suffix_out'])
    
    # with open(partialRadical + 'history_twoPoints_dispBound' + Run_id + '.dat', 'wb') as f:
    #     pickle.dump(history.history, f)
        
    model.save_weights(fnames['prefix_out'] + 'weights' + fnames['suffix_out'])
    
    return history

run_id = 48

folder = ["/Users", "/home"][1] + "/felipefr/switchdrive/scratch/deepBoundary/smartGeneration/LHS_frozen_p4_volFraction/"
nameSnaps = folder + 'snapshots_{0}.h5'
nameC = folder + 'Cnew.h5'
nameMeshRefBnd = 'boundaryMesh.xdmf'
nameWbasis = folder + 'Wbasis_new.h5'
nameYlist = folder + 'Y.h5'
nameTau = folder + 'tau2.h5'
nameEllipseData = folder + 'ellipseData_1.h5'

nX = 36 # because the only first 4 are relevant, the other are constant
nY = 40 # 10 alphas

nsTrain = 10240

# net = {'Neurons': 6*[1024], 'drps': 8*[0.2], 'activations': ['relu','relu','relu'], 
#        'reg': 1.0e-5, 'lr': 0.0001, 'decay' : 1.0, 'epochs': 200} # normally reg = 1e-5

# 48 , p4_volFraction ==> Adam, complete, new loss, 0.2 validation
net = {'Neurons': 5*[512], 'drps': 7*[0.2], 'activations': ['relu','relu','sigmoid'], 
        'reg': 1.0e-5, 'lr': 1.0e-5, 'decay' : 1.0, 'epochs': 500} # normally reg = 1e-5

# 47 , p4_volFraction ==> Adam, complete, new loss, 0.2 validation
# net = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','relu'], 
#         'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 500} # normally reg = 1e-5


# 42 , p4_volFraction ==> Adam, complete, new loss, 0.2 validation
# net = {'Neurons': 6*[1024], 'drps': 8*[0.2], 'activations': ['relu','relu','relu'], 
#        'reg': 1.0e-5, 'lr': 0.0001, 'decay' : 1.0, 'epochs': 200} # normally reg = 1e-5

# 41 , p2_volFraction ==> Adam, complete, new loss
# net = {'Neurons': 6*[1024], 'drps': 8*[0.2], 'activations': ['relu','relu','relu'], 
#        'reg': 1.0e-5, 'lr': 0.0001, 'decay' : 1.0, 'epochs': 200} # normally reg = 1e-5

# 40 , p3_volFraction ==> Adam, complete, new loss
    # net = {'Neurons': 6*[1024], 'drps': 8*[0.2], 'activations': ['relu','relu','relu'], 
    #        'reg': 1.0e-5, 'lr': 0.0001, 'decay' : 1.0, 'epochs': 100} # normally reg = 1e-5

# 39 , p2_volFraction ==> Adam, complete, new loss
    # net = {'Neurons': 6*[1024], 'drps': 8*[0.2], 'activations': ['relu','relu','relu'], 
    #        'reg': 1.0e-5, 'lr': 0.0001, 'decay' : 1.0, 'epochs': 100} # normally reg = 1e-5

# 38 , p4_volFraction ==> Adam, complete, new loss
    # net = {'Neurons': 6*[1024], 'drps': 8*[0.2], 'activations': ['relu','relu','relu'], 
    #        'reg': 1.0e-5, 'lr': 0.0001, 'decay' : 1.0, 'epochs': 100} # normally reg = 1e-5

# 37 , p4_volFraction ==> Adam, complete, new loss
    # net = {'Neurons': 6*[1024], 'drps': 8*[0.2], 'activations': ['relu','relu','relu'], 
    #        'reg': 1.0e-5, 'lr': 0.0001, 'decay' : 1.0, 'epochs': 100} # normally reg = 1e-5
    
# 36 , p4_volFraction ==> Adam, complete
    # net = {'Neurons': 6*[1024], 'drps': 8*[0.2], 'activations': ['relu','relu','relu'], 
    #        'reg': 1.0e-5, 'lr': 0.0001, 'decay' : 1.0, 'epochs': 100} # normally reg = 1e-5
    
# 36 , p4_volFraction ==> Adam, complete
    # net = {'Neurons': 6*[1024], 'drps': 8*[0.2], 'activations': ['relu','relu','relu'], 
    #        'reg': 1.0e-5, 'lr': 0.0001, 'decay' : 1.0, 'epochs': 100} # normally reg = 1e-5

# 35 , p4_volFraction ==> Adam
    # net = {'Neurons': 6*[1024], 'drps': 8*[0.2], 'activations': ['relu','relu','relu'], 
    #        'reg': 1.0e-5, 'lr': 0.0001, 'decay' : 0.5, 'epochs': 100} # normally reg = 1e-5

# 34 , p4_volFraction ==> Adam
# net = {'Neurons': 6*[1024], 'drps': 8*[0.2], 'activations': ['relu','relu','relu'], 
#         'reg': 1.0e-6, 'lr': 0.0001, 'decay' : 0.5, 'epochs': 100} # normally reg = 1e-5

# 33 , p4_volFraction ==> Adam
# net = {'Neurons': 6*[1024], 'drps': 8*[0.2], 'activations': ['relu','relu','relu'], 
       # 'reg': 1.0e-6, 'lr': 0.001, 'decay' : 0.5, 'epochs': 100} # normally reg = 1e-5


# 32 , p4_volFraction ==> RMSprop
# net = {'Neurons': 6*[1024], 'drps': 8*[0.2], 'activations': ['relu','relu','relu'], 
       # 'reg': 1.0e-6, 'lr': 0.001, 'decay' : 0.5, 'epochs': 100} # normally reg = 1e-5

# 30 , p2_volFraction ==> RMSprop
# net = {'Neurons': 6*[1024], 'drps': 8*[0.2], 'activations': ['relu','relu','relu'], 
       # 'reg': 1.0e-6, 'lr': 0.001, 'decay' : 0.5, 'epochs': 100} # normally reg = 1e-5

# 29, p2_volFraction
# net = {'Neurons': 6*[1024], 'drps': 8*[0.2], 'activations': ['relu','relu','relu'], 
#        'reg': 1.0e-6, 'lr': 0.01, 'decay' : 0.5, 'epochs': 500} # normally reg = 1e-5

# 28
# net = {'Neurons': 6*[1024], 'drps': 8*[0.2], 'activations': ['relu','relu','relu'], 
#        'reg': 1.0e-6, 'lr': 0.01, 'decay' : 0.5, 'epochs': 500} # normally reg = 1e-5

# 27
# net = {'Neurons': 6*[1024], 'drps': 8*[0.2], 'activations': ['relu','relu','relu'], 
#        'reg': 1.0e-6, 'lr': 0.01, 'decay' : 0.5, 'epochs': 100} # normally reg = 1e-5
       
fnames = {}      
fnames['suffix_out'] = '_LHS_p4_volFraction_drop02_nX{0}_nY{1}_{2}'.format(nX,nY,run_id)
fnames['prefix_out'] = 'pabloIdea/'
fnames['prefix_in_X'] = folder + "ellipseData_1.h5"
fnames['prefix_in_Y'] = folder + "Y.h5"

os.system("mkdir " + fnames['prefix_out']) # in case the folder is not present

start = timer()
weight = 0.75**np.arange(1,41)/np.sum(0.75**np.arange(1,41))
weight = weight.astype('float32')
hist = basicModelTraining(nsTrain, nX, nY, net, fnames, w_l = weight)
end = timer()

print(end - start)

plt.figure(1)
plt.plot(hist.history['mse'])
plt.plot(hist.history['val_mse'])
plt.grid()
plt.yscale('log')
# plt.savefig(fnames['prefix_out'] + '/mse_46.png')
plt.show()


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

