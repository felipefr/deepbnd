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
        # X_bound = np.array(f['Unique/U_bound'][ns_start:ns_end,:]) # displacements
        X_int = np.array(f['Unique/U_interior'][ns_start:ns_end,:]) # displacements
        Y = np.array(f['Unique/S'][ns_start:ns_end,:]) # stesses
        
    
    Y = np.stack( tuple( [np.mean(Y[:,i:30:3],axis=1) for i in range(3)] ), axis = 1 ) # getting the mean of the minimum stress and its position
    X = X_int
    
    
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

#1) Neurons= 4*[64] , drps = 6*[0.0] , lr2 = 0.00001,  lr = 0.001, decay = 0.1, w_l = 1.0,  param

Neurons= 4*[64]
drps = 6*[0.0]
lr2 = 0.00001

mytf.tf.set_random_seed(4)

model = mytf.DNNmodel(Nin, Nout, Neurons, actLabel = ['relu','relu','relu'], drps = drps, lambReg = lr2  )


history = mytf.my_train_model( model, X, Y, num_parameters, EPOCHS, lr = 0.001, decay = 0.1, w_l = 1.0, w_mu = 0.0)
    
mytf.plot_history( history, savefile = partialRadical + 'plot_history_oneMinima_disp.png')

with open(partialRadical + 'history_oneMinima_disp' + Run_id + '.dat', 'wb') as f:
    pickle.dump(history.history, f)
    
model.save_weights(partialRadical + 'weights_oneMinima_disp' + Run_id)



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

j = 0 
mpt.visualiseStresses9x9(scalerY.inverse_transform(Y_t[j:j+9,:]),scalerY.inverse_transform(Y_p[j:j+9,:]), 
                      figNum = j, savefig = partialRadical + '_oneMinimal_disp_{0}.png'.format(Run_id))


print(error)
with open(partialRadical + '_oneMinima_disp_error_{0}.json'.format(Run_id), 'w') as file:
     file.write(json.dumps(error)) # use `json.loads` to do the reverse


mpt.visualiseScatterErrors(Y_t, Y_p, ['stress Min','x Min','y Min'], gamma = 1.0, 
                           figNum = 7, savefig = partialRadical + '_scatterError_oneMinima_disp_{0}.png'.format(Run_id))

plt.show()
    