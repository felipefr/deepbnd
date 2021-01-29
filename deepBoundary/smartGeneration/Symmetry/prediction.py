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

def getTraining(ns_start, ns_end, nX, nY, Xdatafile, Ydatafile, scalerX = None, scalerY = None):
    X = np.zeros((ns_end - ns_start,nX))
    Y = np.zeros((ns_end - ns_start,nY))
    
    X = myhd.loadhd5(Xdatafile, 'ellipseData')[ns_start:ns_end,:nX,2]
    Y = myhd.loadhd5(Ydatafile, 'Ylist')[ns_start:ns_end,:nY]
    
    XT1 = np.zeros((ns_end - ns_start,nX))
    XT2 = np.zeros((ns_end - ns_start,nX))
    XT3 = np.zeros((ns_end - ns_start,nX))

    for i in range(ns_end - ns_start):
        XT1[i,:] = mirrorHorizontal(X[i,:])
        XT2[i,:] = mirrorVertical(X[i,:])
        XT3[i,:] = mirrorDiagonal(X[i,:])
    
    YdatafileT1 = Ydatafile[:-3] + '_T1' + Ydatafile[-3:]
    YdatafileT2 = Ydatafile[:-3] + '_T2' + Ydatafile[-3:]
    YdatafileT3 = Ydatafile[:-3] + '_T3' + Ydatafile[-3:]
    
    print(YdatafileT1)
    YT1 = myhd.loadhd5(YdatafileT1, 'Ylist')[ns_start:ns_end,:nY]
    YT2 = myhd.loadhd5(YdatafileT2, 'Ylist')[ns_start:ns_end,:nY]
    YT3 = myhd.loadhd5(YdatafileT3, 'Ylist')[ns_start:ns_end,:nY]
    
    X = np.concatenate((X,XT1,XT2,XT3))
    Y = np.concatenate((Y,YT1,YT2,YT3))
    
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


folder = rootData + "/deepBoundary/smartGeneration/LHS_frozen_p4_volFraction/"
nameSnaps = folder + 'snapshots_{0}.h5'
nameC = folder + 'Cnew.h5'
nameMeshRefBnd = 'boundaryMesh.xdmf'
nameWbasis = folder + 'Wbasis_new.h5'
nameYlist = folder + 'Y.h5'
nameTau = folder + 'tau2.h5'
nameEllipseData = folder + 'ellipseData_1.h5'

nX = 36 # because the only first 4 are relevant, the other are constant
nY = 140 # 10 alphas

nsTrain = 10240

# run_id = sys.argv[1]
run_id = 1
print("run_id is " ,  run_id)

nets = {}

# tests
nets['1'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
    'reg': 0.0, 'lr': 1.0e-3, 'decay' : 1.0, 'epochs': 1000} # normally reg = 1e-5

nets['2'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000} # normally reg = 1e-5

nets['3'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-5, 'decay' : 1.0, 'epochs': 1000} # normally reg = 1e-5

nets['4'] = {'Neurons': 5*[50], 'drps': 7*[0.2], 'activations': ['relu','relu','linear'], 
        'reg': 1.0e-4, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000} # normally reg = 1e-5

nets['5'] = {'Neurons': 5*[100], 'drps': 7*[0.2], 'activations': ['relu','relu','linear'], 
        'reg': 1.0e-4, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000} # normally reg = 1e-5

nets['6'] = {'Neurons': 5*[200], 'drps': 7*[0.2], 'activations': ['relu','relu','linear'], 
        'reg': 1.0e-4, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000} # normally reg = 1e-5

# First Arch : U curve
nets['7'] = {'Neurons': 5*[200], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000} # normally reg = 1e-5

nets['8'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000, 'weightTsh' : 5} # normally reg = 1e-5

nets['9'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000, 'weightTsh' : 10} # normally reg = 1e-5

nets['10'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000, 'weightTsh' : 15} # normally reg = 1e-5

nets['11'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000, 'weightTsh' : 20} # normally reg = 1e-5

nets['12'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000, 'weightTsh' : 25} # normally reg = 1e-5

nets['13'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000, 'weightTsh' : 30} # normally reg = 1e-5

nets['14'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000, 'weightTsh' : 35} # normally reg = 1e-5

# Second Arch : U curve
nets['15'] = {'Neurons': 5*[200], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000, 'weightTsh' : 5} # normally reg = 1e-5

nets['16'] = {'Neurons': 5*[200], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000, 'weightTsh' : 15} # normally reg = 1e-5

nets['17'] = {'Neurons': 5*[200], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000, 'weightTsh' : 25} # normally reg = 1e-5

nets['18'] = {'Neurons': 5*[200], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000, 'weightTsh' : 35} # normally reg = 1e-5

# Third Arch : U curve
nets['19'] = {'Neurons': 5*[200], 'drps': 7*[0.1], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000, 'weightTsh' : 5} # normally reg = 1e-5

nets['20'] = {'Neurons': 5*[200], 'drps': 7*[0.1], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000, 'weightTsh' : 15} # normally reg = 1e-5

nets['21'] = {'Neurons': 5*[200], 'drps': 7*[0.1], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000, 'weightTsh' : 25} # normally reg = 1e-5

nets['22'] = {'Neurons': 5*[200], 'drps': 7*[0.1], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000, 'weightTsh' : 35} # normally reg = 1e-5

nets['23'] = {'Neurons': 5*[200], 'drps': 7*[0.1], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000, 'weightTsh' : 40} # normally reg = 1e-5

# Fourth Arch : U curve
nets['24'] = {'Neurons': 6*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000, 'weightTsh' : 40} # normally reg = 1e-5

nets['25'] = {'Neurons': 4*[200], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000, 'weightTsh' : 40} # normally reg = 1e-5

nets['26'] = {'Neurons': 6*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000, 'weightTsh' : 5} # normally reg = 1e-5

nets['27'] = {'Neurons': 4*[200], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000, 'weightTsh' : 5} # normally reg = 1e-5

nets['28'] = {'Neurons': 5*[200], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 1.0e-5, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000, 'weightTsh' : 40} # normally reg = 1e-5

nets['29'] = {'Neurons': 5*[200], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 1.0e-5, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000, 'weightTsh' : 5} # normally reg = 1e-5

# series with more modes

nets['30'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000, 'weightTsh' : 60} # normally reg = 1e-5

nets['31'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000, 'weightTsh' : 80} # normally reg = 1e-5

nets['32'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000, 'weightTsh' : 100} # normally reg = 1e-5

nets['33'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000, 'weightTsh' : 120} # normally reg = 1e-5

nets['34'] = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-4, 'decay' : 1.0, 'epochs': 1000, 'weightTsh' : 140} # normally reg = 1e-5

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


net = nets[str(run_id)]

fnames = {}      
fnames['suffix_out'] = '_LHS_p4_volFraction_drop02_nX{0}_nY{1}_{2}'.format(nX,nY,run_id)
fnames['prefix_out'] = rootData + '/deepBoundary/smartGeneration/newTrainingSymmetry/'
fnames['prefix_in_X'] = folder + "ellipseData_1.h5"
fnames['prefix_in_Y'] = folder + "Y.h5"

# 'weights_LHS_p4_volFraction_drop02_nX36_nY140_42'

os.system("mkdir " + fnames['prefix_out']) # in case the folder is not present

start = timer()

# mytf.DNNmodel(nX, nY, Neurons, actLabel = actLabel , drps = drps, lambReg = reg  )

X, Y, scalerX, scalerY = getTraining(0,nsTrain, nX, nY, fnames['prefix_in_X'], fnames['prefix_in_Y'])

models = []
for j in range(35,49):
    net =  nets[str(j)]
    fnames['suffix_out'] = '_{0}_{1}'.format(nY,j)        
    models.append(mytf.DNNmodel(nX, nY, net['Neurons'], actLabel = net['activations'], drps = net['drps'], lambReg = net['reg']))
    print(fnames['prefix_out'] + 'weights' + fnames['suffix_out'])
    models[-1].load_weights( fnames['prefix_out'] + 'weights_LHS_p4_volFraction_drop02_nX36_nY140_{0}'.format(j))
        
Y_p = models[12].predict(X[:,:])

plt.figure(1,(12,9))
plt.suptitle('Model nY = 140')
N0 = 134
for i in range(6):
    plt.subplot('23' + str(i+1))
    plt.title('Alpha ' + str(i+1 + N0))
    plt.scatter(Y[:,i + N0], Y_p[:,N0 + i],marker = '.')
    plt.plot([0.0,1.0],[0.0,1.0],'k-')
    plt.ylabel('Y reference')
    plt.xlabel('Y predicted')
    plt.grid()

# plt.tight_layout(pad = 1.09)
plt.savefig('Scatter_model_nY140_lasts2.png')
plt.show()

# hist = basicModelTraining(nsTrain, nX, nY, net, fnames, w_l = -1.0)
end = timer()

# print(end - start)

# plt.figure(2)
# plt.plot(hist.history['mse'])
# plt.plot(hist.history['val_mse'])
# plt.grid()
# plt.yscale('log')
# plt.savefig(fnames['prefix_out'] + '/plot_mse_{0}.png'.format(run_id))
# plt.show()




