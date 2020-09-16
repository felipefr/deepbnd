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
from sklearn.processing import MinMaxScaler







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

ns = 1000

with h5py.File(radical + 'dataset.hdf5', 'r') as f, h5py.File(radical + 'ParamFile.hdf5', 'r') as g:
    X_train_param = np.array(g['Unique/sample'][:ns,:])  # displacements + param
    X_train_bound = np.array(f['Unique/U_bound'][:ns,:]) # displacements
    X_train_int = np.array(f['Unique/U_interior'][:ns,:]) # displacements
    Y_train = np.array(f['Unique/S'][:ns,:]) # stesses
    

Y_train = np.stack( tuple( [np.mean(Y_train[:,i::3],axis=1) for i in range(3)] ) ) # getting the mean of the minimum stress and its position

scalerY = MinMaxScaler()
scalerY.fit(Y_train)
Y_t = scaler.transform(Y_train)

X_train = X_train_param
scalerX = MinMaxScaler()
scalerX.fit(X_train)
X_t = scalerX.transform(X_train)

Nin = X_t.shape[1]
Nout = Y_t.shape[1]

mytf.tf.set_random_seed(3)

Run_id = '1'
EPOCHS = 30

num_parameters = 0 # parameters that are taking into account into the model for minimization (split the norm)

start = timer()

Neurons= [1024,512,256,64]
drps =6*[0.0]
lr2 = 0.0
model = mytf.DNNmodel(Nin, Nout, Neurons, actLabel = ['relu','relu','relu'], drps = drps, lambReg = lr2  )

history = mytf.my_train_model( model, X_t, Y_t, num_parameters, EPOCHS, lr = 0.001, decay = 0.1, w_l = 1.0, w_mu = 0.0)
    
mytf.plot_history( history)

with open(radicalPartial + 'history_' + Run_id + '.dat', 'wb') as f:
    pickle.dump(history.history, f)
    
model.save_weights(radicalPartial + 'weights_' + Run_id)

end = timer()
print('time', end - start) 

# Prediction step
# ntest = 10
# with h5py.File(radical + 'dataset.hdf5', 'r') as f, h5py.File(radical + 'ParamFile.hdf5', 'r') as g:
#     X_test = np.concatenate( ( np.array(f['Unique/X'][ns:ns+ntest,:]), np.array(g['Unique/sample'][ns:ns+ntest,:]) ), axis = 1) # displacements
#     Y_test = np.array(f['Unique/Y'][ns:ns+ntest,:]) # stesses


# ind = np.concatenate((np.arange(0,20),np.arange(40,X_test.shape[1]))).astype('int')
# X_test = X_test[:, ind] # this is because I generated wrongly, including bottom displacements

# X_test_t = TX.transform(X_test)
# Y_pred_t = model.predict(X_test_t)
# Y_pred = TY.inverse_transform(Y_pred_t)

# indStress = np.arange(0,60,3)
# indStressMin = np.arange(0,30,3)
# indStressMax = np.arange(30,60,3)
# indx =  np.arange(1,60,3)
# indy =  np.arange(2,60,3)
# indxMin =  np.arange(1,30,3)
# indyMin =  np.arange(2,30,3)
# indxMax =  np.arange(31,60,3)
# indyMax =  np.arange(32,60,3)


# print(np.linalg.norm(Y_pred - Y_test)/ntest)
# print(np.linalg.norm(Y_pred[:,indStress] - Y_test[:,indStress])/ntest)
# print(np.linalg.norm(Y_pred[:,indx] - Y_test[:,indx])/ntest)
# print(np.linalg.norm(Y_pred[:,indy] - Y_test[:,indy])/ntest)
# print(np.linalg.norm(Y_pred - Y_test, axis = 0)/ntest)

# print(np.linalg.norm(Y_pred - Y_test)/np.linalg.norm(Y_test))
# print(np.linalg.norm(Y_pred[:,indStress] - Y_test[:,indStress])/np.linalg.norm(Y_test[:,indStress]))
# print(np.linalg.norm(Y_pred[:,indx] - Y_test[:,indx])/np.linalg.norm(Y_test[:,indx]))
# print(np.linalg.norm(Y_pred[:,indy] - Y_test[:,indy])/np.linalg.norm(Y_test[:,indy]))
# print(np.linalg.norm(Y_pred - Y_test, axis = 0)/np.linalg.norm(Y_test,axis = 0))

