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

base_offline_folder = '.'
folderData = '../generationDataEllipseEx/simuls2/'

# filesToMerge = [ folderData + 'prob2_{0}ParamFile.hdf5'.format(i) for i in range(3)]
# myHDF5.merge(filesToMerge, 'prob2_ParamFile.hdf5', ['Unique/sample'], ['Unique/sample'], mode = 'w')
# myHDF5.merge(filesToMerge, 'prob2_dataset.hdf5', ['Unique/Y'], ['Unique/Y'], mode = 'r+')


ns = 1000

with h5py.File('prob2_dataset.hdf5', 'r') as f, h5py.File('prob2_ParamFile.hdf5', 'r') as g:
    X_train = np.concatenate( ( np.array(f['Unique/X'][:ns,:]), np.array(g['Unique/sample'][:ns,:]) ), axis = 1) # displacements
    Y_train = np.array(f['Unique/Y'][:ns,:]) # stesses
    
    
ind = np.concatenate((np.arange(0,20),np.arange(40,X_train.shape[1]))).astype('int')
X_train = X_train[:, ind] # this is because I generated wrongly, including bottom displacements

# Obs: I've checked that pca space is well defined

TY = dman.myTransfomation(Y_train, 'minmax') 
TY.registerSubset('stress' , np.arange(0,60,3) )
TY.registerSubset('x' , np.sort(np.concatenate((np.arange(1,60,3), np.arange(2,60,3)))) )
Y_t = TY.transform(Y_train)
# TY.showStats()
# TY.showStats(Y_t)

TX = dman.myTransfomation(X_train, 'minmax') 
TX.registerSubset('u' , np.arange(0,40) )
[ TX.registerSubset('p' + str(i) , [40 + i]) for i in range(5)] 
X_t = TX.transform(X_train)
# TX.showStats()
# TX.showStats(X_t)

Nin = X_t.shape[1]
Nout = Y_t.shape[1]

mytf.tf.set_random_seed(3)

Run_id = ''
EPOCHS = 30

num_parameters = 5 # parameters that are taking into account into the model for minimization (split the norm)

start = timer()

Neurons= 3*[64]
drps =[0.0,0.2,0.2,0.2,0.0]
lr2 = 0.00001
model = mytf.DNNmodel(Nin, Nout, Neurons, actLabel = ['relu','relu','relu'], drps = drps, lambReg = lr2  )

history = mytf.my_train_model( model, X_t, Y_t, num_parameters, EPOCHS, lr = 0.01, decay = 0.1, w_l = 0.5, w_mu = 0.5)
    
mytf.plot_history( history)

with open('prob2_trivial_history_1' + Run_id + '.dat', 'wb') as f:
    pickle.dump(history.history, f)
    
model.save_weights('prob2_trivial_weights_1' + Run_id)

end = timer()
print('time', end - start) 

# Prediction step
ntest = 10
with h5py.File('prob2_dataset.hdf5', 'r') as f, h5py.File('prob2_ParamFile.hdf5', 'r') as g:
    X_test = np.concatenate( ( np.array(f['Unique/X'][ns:ns+ntest,:]), np.array(g['Unique/sample'][ns:ns+ntest,:]) ), axis = 1) # displacements
    Y_test = np.array(f['Unique/Y'][ns:ns+ntest,:]) # stesses


ind = np.concatenate((np.arange(0,20),np.arange(40,X_test.shape[1]))).astype('int')
X_test = X_test[:, ind] # this is because I generated wrongly, including bottom displacements

X_test_t = TX.transform(X_test)
Y_pred_t = model.predict(X_test_t)
Y_pred = TY.inverse_transform(Y_pred_t)

indStress = np.arange(0,60,3)
indStressMin = np.arange(0,30,3)
indStressMax = np.arange(30,60,3)
indx =  np.arange(1,60,3)
indy =  np.arange(2,60,3)
indxMin =  np.arange(1,30,3)
indyMin =  np.arange(2,30,3)
indxMax =  np.arange(31,60,3)
indyMax =  np.arange(32,60,3)


print(np.linalg.norm(Y_pred - Y_test)/ntest)
print(np.linalg.norm(Y_pred[:,indStress] - Y_test[:,indStress])/ntest)
print(np.linalg.norm(Y_pred[:,indx] - Y_test[:,indx])/ntest)
print(np.linalg.norm(Y_pred[:,indy] - Y_test[:,indy])/ntest)
print(np.linalg.norm(Y_pred - Y_test, axis = 0)/ntest)

print(np.linalg.norm(Y_pred - Y_test)/np.linalg.norm(Y_test))
print(np.linalg.norm(Y_pred[:,indStress] - Y_test[:,indStress])/np.linalg.norm(Y_test[:,indStress]))
print(np.linalg.norm(Y_pred[:,indx] - Y_test[:,indx])/np.linalg.norm(Y_test[:,indx]))
print(np.linalg.norm(Y_pred[:,indy] - Y_test[:,indy])/np.linalg.norm(Y_test[:,indy]))
print(np.linalg.norm(Y_pred - Y_test, axis = 0)/np.linalg.norm(Y_test,axis = 0))

plt.figure(1,(15,15))
plt.subplot('221')
plt.title('Value Min Stress')
plt.imshow(Y_test[:,indStressMin])
plt.colorbar()

plt.subplot('222')
plt.title('Value Max Stress')
plt.imshow(Y_test[:,indStressMax])
plt.colorbar()

plt.subplot('223')
plt.title('relative error Min Stress')
plt.imshow(np.abs((Y_test-Y_pred)/Y_test)[:,indStressMin])
plt.colorbar()

plt.subplot('224')
plt.title('relative error Max Stress')
plt.imshow(np.abs((Y_test-Y_pred)/Y_test)[:,indStressMax])
plt.colorbar()
plt.plot()

plt.figure(2,(15,15))
plt.subplot('221')
plt.title('Value Min X')
plt.imshow(Y_test[:,indxMin])
plt.colorbar()

plt.subplot('222')
plt.title('Value Max X')
plt.imshow(Y_test[:,indxMax])
plt.colorbar()

plt.subplot('223')
plt.title('abs error Min X')
plt.imshow(np.abs(Y_test-Y_pred)[:,indxMin])
plt.colorbar()

plt.subplot('224')
plt.title('abs error Max X')
plt.imshow(np.abs((Y_test-Y_pred)/Y_test)[:,indxMax])
plt.colorbar()
plt.plot()


plt.figure(3,(15,15))
plt.subplot('221')
plt.title('Value Min Y')
plt.imshow(Y_test[:,indyMin])
plt.colorbar()

plt.subplot('222')
plt.title('Value Max Y')
plt.imshow(Y_test[:,indyMax])
plt.colorbar()

plt.subplot('223')
plt.title('abs error Min Y')
plt.imshow(np.abs(Y_test-Y_pred)[:,indyMin])
plt.colorbar()

plt.subplot('224')
plt.title('abs error Max Y')
plt.imshow(np.abs((Y_test-Y_pred)/Y_test)[:,indyMax])
plt.colorbar()
plt.plot()

plt.show()