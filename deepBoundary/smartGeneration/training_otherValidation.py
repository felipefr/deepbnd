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

def getTraining(ns , nX, nY, Xdatafile, Ydatafile, scalerX = None, scalerY = None):
    Xlist = []
    Ylist = []
        
    ndata = len(Xdatafile)

    for Xdatafile_i, Ydatafile_i in zip(Xdatafile,Ydatafile):    
        Xlist.append(myhd.loadhd5(Xdatafile_i, 'ellipseData')[:,:nX,2])
        Ylist.append(myhd.loadhd5(Ydatafile_i, 'Ylist')[:,:nY])
    
    X = np.concatenate(tuple(Xlist),axis = 0)
    Y = np.concatenate(tuple(Ylist),axis = 0)
    
    if(type(scalerX) == type(None)):
        scalerX = MinMaxScaler()
        scalerX.fit(X)
    
    if(type(scalerY) == type(None)):
        scalerY = MinMaxScaler()
        scalerY.fit(Y)
            
    return scalerX.transform(X), scalerY.transform(Y), scalerX, scalerY


def basicModelTraining(ns, nX, nY, net, fnames, w_l = 1.0, ratio_val = 0.2):
    X, Y, scalerX, scalerY = getTraining(ns, nX, nY, fnames['prefix_in_X'], fnames['prefix_in_Y'])
    
    Neurons = net['Neurons']
    actLabel = net['activations']
    drps = net['drps']
    reg = net['reg']
    Epochs = net['epochs']
    decay = net['decay']
    lr = net['lr']
    
    # suffle only training
    nshuffle = int((1.0 - ratio_val)*ns)
    
    indices = np.arange(len(X))
    np.random.shuffle(indices[:nshuffle])
    np.random.shuffle(indices[nshuffle:])

    X = X[indices,:]
    Y = Y[indices,:]
    
    if(w_l<0.0): # to indicate we want to compute problem-specific weights
        maxAlpha = scalerY.data_max_
        minAlpha = scalerY.data_min_
        
        w_l = (maxAlpha - minAlpha)**2.0
        w_l = w_l.astype('float32')
    
    model = mytf.DNNmodel(nX, nY, Neurons, actLabel = actLabel , drps = drps, lambReg = reg  )
    
    num_parameters = 0 # in case of treating differently a part of the outputs
    history = mytf.my_train_model( model, X, Y, num_parameters, Epochs, lr = lr, 
                                  decay = decay, w_l = w_l, w_mu = 0.0, ratio_val = ratio_val)
        
    mytf.plot_history( history, savefile = fnames['prefix_out'] + 'plot_history' + fnames['suffix_out'])
    
    # with open(partialRadical + 'history_twoPoints_dispBound' + Run_id + '.dat', 'wb') as f:
    #     pickle.dump(history.history, f)
        
    model.save_weights(fnames['prefix_out'] + 'weights' + fnames['suffix_out'])
    
    return history

# run_id = 2 # "p4, no validation separated, shuffle, nY = 40"
# run_id = 3 # "p4, no validation separated, shuffle, nY = 40, 500 epochs"

# run_id = 4 # "p3, no validation separated, shuffle, nY = 40, 500 epochs"
# run_id = 5 # "p2, no validation separated, shuffle, nY = 40, 500 epochs"
# run_id = 6 # "minmax, no validation separated, shuffle, nY = 40, 500 epochs"

# run_id = 7 # "p4, no validation separated, shuffle, 100 neurons, nY = 40, 2000 epochs"
# run_id = 8 # "p4, no validation separated, shuffle, 100 neurons, nY = 40, 500 epochs, new loss"
run_id = 9 # "p4, no validation separated, shuffle, 100 neurons, nY = 40, 500 epochs, new loss"

f = open("../../../rootDataPath.txt")
rootData = f.read()[:-1]
f.close()

folderTrain = "{0}/deepBoundary/smartGeneration/LHS_frozen_p4_volFraction/".format(rootData)
folderVal = rootData + "/deepBoundary/smartGeneration/validation_and_test/"

print(folderTrain, rootData)

nameMeshRefBnd = 'boundaryMesh.xdmf'

nX = 36 
nY = 40 # 10 alphas

nsTrain = 10240
nsval = 2000
ns = nsval + nsTrain
ratio_val = nsval/ns

# 48 , p4_volFraction ==> Adam, complete, new loss, 0.2 validation
net = {'Neurons': 5*[100], 'drps': 7*[0.0], 'activations': ['relu','relu','sigmoid'], 
        'reg': 0.0, 'lr': 1.0e-5, 'decay' : 1.0, 'epochs': 10} # normally reg = 1e-5
       
fnames = {}      
fnames['suffix_out'] = '_p4_nX{0}_nY{1}_{2}'.format(nX,nY,run_id)
fnames['prefix_out'] = rootData + '/deepBoundary/smartGeneration/differentModels/'
fnames['prefix_in_X'] = [folderTrain + "ellipseData_1.h5", folderVal + "ellipseData_validation.h5"]
fnames['prefix_in_Y'] = [folderTrain + "Y.h5", folderVal + "Y_validation_p4.h5"]
# fnames['prefix_in_X'] = [folderTrain + "ellipseData_1.h5"]
# fnames['prefix_in_Y'] = [folderTrain + "Y.h5"]

os.system("mkdir " + fnames['prefix_out']) # in case the folder is not present

start = timer()

# weight = np.ones(nY)
hist = basicModelTraining(ns, nX, nY, net, fnames, w_l = -1.0 , ratio_val = ratio_val)
historyArray = np.array([hist.history[k] for k in ['loss','val_loss','mse','val_mse', 'mae', 'val_mae']])
np.savetxt(fnames['prefix_out'] + 'history' + fnames['suffix_out'] + '.txt', historyArray)
end = timer()

print(end - start)

plt.figure(1)
plt.plot(hist.history['mse'] , label = 'train')
plt.plot(hist.history['val_mse'], label = 'validation')
plt.legend()
plt.grid()
plt.yscale('log')
plt.ylabel('mse')
plt.xlabel('epochs')
plt.savefig(fnames['prefix_out'] + '/plot_mse_{0}.png'.format(run_id))


plt.figure(2)
plt.plot(hist.history['mae'] , label = 'train')
plt.plot(hist.history['val_mae'], label = 'validation')
plt.legend()
plt.grid()
plt.yscale('log')
plt.ylabel('mae')
plt.xlabel('epochs')
plt.savefig(fnames['prefix_out'] + '/plot_mae_{0}.png'.format(run_id))


# plt.show()


# 1

