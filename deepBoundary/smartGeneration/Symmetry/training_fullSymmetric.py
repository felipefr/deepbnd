import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0,'../')
sys.path.insert(0,'../../../utils/')
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
import copy

def mirror(X, perm):
    perm = list(np.array(perm) - 1)
    X2 = copy.deepcopy(X)
    X2[:] = X2[perm]
    return X2

def mirrorHorizontal(X):
    perm = [2,1,4,3,8,7,6,5,10,9,12,11,16,15,14,13,22,21,20,19,18,17,24,23,26,25,28,27,30,29,36,35,34,33,32,31]
    return mirror(X,perm)

def mirrorVertical(X):
    perm = [3,4,1,2,13,14,15,16,11,12,9,10,5,6,7,8,31,32,33,34,35,36,29,30,27,28,25,26,23,24,17,18,19,20,21,22]
    return mirror(X,perm)

def mirrorDiagonal(X):
    return mirrorVertical(mirrorHorizontal(X))
    
def createSymmetricEllipseData(ellipseFileName, ellipseFileName_new):
    EllipseData = myhd.loadhd5(ellipseFileName, 'ellipseData')
    X = EllipseData[:,:,2]
    
    ns0 = len(X)
    nX = len(X[0])
    
    EllipseData_list = [EllipseData] 
    for mirror_func in [mirrorHorizontal, mirrorVertical, mirrorDiagonal]: 
        EllipseData_list.append(copy.deepcopy(EllipseData))
        for i in range(ns0):
            EllipseData_list[-1][i,:,2] = mirror_func(X[i,:])
    
    EllipseData_new = np.concatenate(tuple(EllipseData_list))
    
    myhd.savehd5(ellipseFileName_new, EllipseData_new, 'ellipseData', mode = 'w' )

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
    lr = net['lr']; weightThreshold = net['weightTsh']
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    X = X[indices,:]
    Y = Y[indices,:]
    
    if(w_l<0.0): # to indicate we want to compute problem-specific weights
        maxAlpha = scalerY.data_max_
        minAlpha = scalerY.data_min_
        
        w_l = (maxAlpha - minAlpha)**2.0
        w_l[weightThreshold:] = 0.0      
        w_l = w_l.astype('float32')

    model = mytf.DNNmodel(nX, nY, Neurons, actLabel = actLabel , drps = drps, lambReg = reg  )
    
    num_parameters = 0 # in case of treating differently a part of the outputs
    history = mytf.my_train_model( model, X, Y, num_parameters, Epochs, lr = lr, decay = decay, w_l = w_l, w_mu = 0.0)
        
    mytf.plot_history( history, savefile = fnames['prefix_out'] + 'plot_history' + fnames['suffix_out'])
    
    # with open(partialRadical + 'history_twoPoints_dispBound' + Run_id + '.dat', 'wb') as f:
    #     pickle.dump(history.history, f)
        
    model.save_weights(fnames['prefix_out'] + 'weights' + fnames['suffix_out'])
    
    return history

f = open("../../../../rootDataPath.txt")
rootData = f.read()[:-1]
f.close()

# run_id = sys.argv[1]
# Nrb = sys.argv[2]
run_id = 0
Nrb = 20
epochs = 500
print("run_id is " ,  run_id, 'Nrb is ', Nrb)

folder = rootData + "/deepBoundary/smartGeneration/LHS_p4_fullSymmetric/"
nameMeshRefBnd = 'boundaryMesh.xdmf'
nameYlist = folder + 'Y_svd_full.h5'
nameEllipseData = folder + 'ellipseData_1.h5'
nameEllipseData_new = folder + 'ellipseData_fullSymmetric.h5'

fnames = {}      
fnames['suffix_out'] = 'fullSymmetric_std_{0}_nY{1}'.format(run_id,Nrb)
fnames['prefix_out'] = rootData + '/deepBoundary/smartGeneration/newTrainingSymmetry/fullSymmetric/'
fnames['prefix_in_X'] = nameEllipseData_new
fnames['prefix_in_Y'] = nameYlist


nX = 36 # because the only first 4 are relevant, the other are constant
nY = 160 # 10 alphas
nsTrain = 4*10240

# createSymmetricEllipseData(nameEllipseData, nameEllipseData_new)
X, Y, scalerX, scalerY = getTraining(0,nsTrain, nX, nY, fnames['prefix_in_X'], fnames['prefix_in_Y'])

nets = {}

# series with 5000 epochs
nets[0] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0} # normally reg = 1e-5

net = nets[run_id]
net['epochs'] = epochs
net['weightTsh'] = Nrb

os.system("mkdir " + fnames['prefix_out']) # in case the folder is not present

start = timer()

hist = basicModelTraining(nsTrain, nX, nY, net, fnames, w_l = -1.0)
end = timer()

plt.figure(2)
plt.plot(hist.history['mse'])
plt.plot(hist.history['val_mse'])
plt.grid()
plt.yscale('log')
plt.savefig(fnames['prefix_out'] + '/plot_mse_{0}.png'.format(run_id))
plt.show()