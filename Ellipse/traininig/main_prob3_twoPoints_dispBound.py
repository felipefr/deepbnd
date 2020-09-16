import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0,'../')
sys.path.insert(0,'../utils/')
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


base_offline_folder = '.'
folderData = '../generationDataEllipseEx/simuls2/'
partialRadical = "prob3_"
radical = folderData + partialRadical

# filesToMerge1 = [ radical + '{0}ParamFile.hdf5'.format(i) for i in range(3)]
# filesToMerge2 = [ radical + '{0}dataset.hdf5'.format(i) for i in range(3)]
# myHDF5.merge(filesToMerge1, radical + 'ParamFile.hdf5', ['Unique/sample'], ['Unique/sample'], mode = 'w')
# myHDF5.merge(filesToMerge2, radical + 'dataset.hdf5', ['Unique/X_dispBound'], ['Unique/U_bound'], mode = 'w')
# myHDF5.merge(filesToMerge2, radical + 'dataset.hdf5', ['Unique/X_interior4'], ['Unique/U_interior'], mode = 'r+')
# myHDF5.merge(filesToMerge2, radical + 'dataset.hdf5', ['Unique/Y'], ['Unique/S'], mode = 'r+')

ns = 10000

def getTraining(ns_start, ns_end, scalerX = None, scalerY = None):
    with h5py.File(radical + 'dataset.hdf5', 'r') as f, h5py.File(radical + 'ParamFile.hdf5', 'r') as g:
        # X_param = np.array(g['Unique/sample'][ns_start:ns_end,:])  # displacements + param
        X_bound = np.array(f['Unique/U_bound'][ns_start:ns_end,:]) # displacements
        # X_int = np.array(f['Unique/U_interior'][ns_start:ns_end,:]) # displacements
        Y = np.array(f['Unique/S'][ns_start:ns_end,:]) # stesses
        
    ns = ns_end - ns_start # local 
    Ymin = np.stack( tuple( [np.mean(Y[:,i:30:3],axis=1) for i in range(3)] ), axis = 1 ) # getting the mean of the minimum stress and its position
    
    xyMax = Y[:,30:].reshape((ns,10,3))[:,:,1:3]
    
    dist = np.array( [np.linalg.norm(xyMax[i,:,:] - Ymin[i,1:3], axis = 1) for i in range(ns)])  # creates a distance matrix relative to the centroid of minimal
    ind = np.where(dist>0.15) # returns i, j lists
    
    Ymax = np.zeros_like(Ymin)
    
    for i in range(ns):
        Ymax[i,:] = np.mean(Y[i,30:].reshape((10,3))[ ind[1][ind[0]==i],:], axis = 0)

    Y = np.concatenate((Ymin,Ymax), axis = 1)
    
    X = X_bound
    if(type(scalerX) == type(None)):
        scalerX = MinMaxScaler()
        scalerX.fit(X)
    
    if(type(scalerY) == type(None)):
        scalerY = MinMaxScaler()
        scalerY.fit(Y)
            
    return scalerX.transform(X), scalerY.transform(Y), scalerX, scalerY

X, Y, scalerX, scalerY = getTraining(0,ns)

Nin = X.shape[1]
Nout = Y.shape[1]

Run_id = '1'
EPOCHS = 500

num_parameters = 0 # parameters that are taking into account into the model for minimization (split the norm)

start = timer()

# Neurons= 4*[64] , drps = 6*[0.0] , lr2 = 0.00001

Neurons= 4*[64]
drps = 6*[0.0]
lr2 = 0.00001

mytf.tf.set_random_seed(4)

model = mytf.DNNmodel(Nin, Nout, Neurons, actLabel = ['relu','relu','relu'], drps = drps, lambReg = lr2  )


history = mytf.my_train_model( model, X, Y, num_parameters, EPOCHS, lr = 0.001, decay = 0.1, w_l = 1.0, w_mu = 0.0)
    
mytf.plot_history( history, savefile = partialRadical + 'plot_history_twoPoints_dispBound.png')

with open(partialRadical + 'history_twoPoints_dispBound' + Run_id + '.dat', 'wb') as f:
    pickle.dump(history.history, f)
    
model.save_weights(partialRadical + 'weights_twoPoints_dispBound' + Run_id)

end = timer()
print('time', end - start) 

# Prediction step
ntest = 2000

X_t, Y_t, d1, d2= getTraining(ns,ns + ntest, scalerX, scalerY) # normalised

Y_p = model.predict(X_t)

error = {}
error["ml2"] = np.linalg.norm(Y_p - Y_t)/ntest
error["rl2"] = np.linalg.norm(Y_p - Y_t)/np.linalg.norm(Y_t)
error["ml2_0"] = list(np.linalg.norm(Y_p - Y_t, axis = 0)/ntest)
error["rl2_0"] = list(np.linalg.norm(Y_p - Y_t, axis = 0)/np.linalg.norm(Y_t,axis = 0))

Y_tt = scalerY.inverse_transform(Y_t)
Y_pp = scalerY.inverse_transform(Y_p)

j = 0 
mpt.visualiseStresses9x9(Y_tt[j:j+9,0:3] , Y_pp[j:j+9,0:3] , 
                      figNum = 1, savefig = partialRadical + '_twoPoints_dispBound_Min_{0}.png'.format(Run_id))

mpt.visualiseStresses9x9( Y_tt[j:j+9,3:] , Y_pp[j:j+9,3:] , 
                      figNum = 2, savefig = partialRadical + '_twoPoints_dispBound_Max_{0}.png'.format(Run_id))


print(error)
with open(partialRadical + '_twoPoints_dispBound_error_{0}.json'.format(Run_id), 'w') as file:
     file.write(json.dumps(error)) # use `json.loads` to do the reverse


mpt.visualiseScatterErrors(Y_t[:,0:3], Y_p[:,0:3], ['stress Min','x Min','y Min'], gamma = 1.0, 
                           figNum = 3, savefig = partialRadical + '_scatterError_twoPoints_dispBound_Min_{0}.png'.format(Run_id))

mpt.visualiseScatterErrors(Y_t[:,3:], Y_p[:,3:], ['stress Max','x Max','y Max'], gamma = 1.0, 
                           figNum = 4, savefig = partialRadical + '_scatterError_twoPoints_dispBound_Max_{0}.png'.format(Run_id))

plt.show()
    