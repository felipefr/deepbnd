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

# filesToMerge = [ folderData + 'prob2_{0}dataset.hdf5'.format(i) for i in range(3)]
# myHDF5.merge(filesToMerge, 'prob2_dataset.hdf5', ['Unique/X'], ['Unique/X'], mode = 'w')
# myHDF5.merge(filesToMerge, 'prob2_dataset.hdf5', ['Unique/Y'], ['Unique/Y'], mode = 'r+')

# input() 
ns = 10000

with h5py.File('prob2_dataset.hdf5', 'r') as f:
    X_train = np.array(f['Unique/X'][:ns,:]) # displacements
    Y_train = np.array(f['Unique/Y'][:ns,:]) # stesses

ind = np.concatenate((np.arange(0,20),np.arange(40,60))).astype('int')
X_train = X_train[:, ind] # this is because I generated wrongly, including bottom

# Obs: I've checked that pca space is well defined

subsets3 = {'stress' : np.arange(0,60,3) , 'x' : np.sort(np.concatenate((np.arange(1,60,3), np.arange(2,60,3))))}
Y_r, TY = dman.getCombinedPCAnormalisation(Y_train, subsets3,  N=59)

subsets2 = {'all' : []}
X_r, TX = dman.getCombinedPCAnormalisation(X_train, subsets2, N=39)

Nin = X_r.shape[1]
Nout = Y_r.shape[1]

mytf.tf.set_random_seed(3)

Run_id = ''
EPOCHS = 500

weight_mu = 0.0

num_parameters = 0 # parameters that are taking into account into the model for minimization.

start = timer()

# good arch
# 1) all 'minmax',  Neurons= [256,256,256,256], 'relu', lr = 0.001, decay = 0.1, PcaX = 25, PcaY = 25, 
#              normalised, reg 0.00001, drps =[0.0,0.1,0.1,0.1,0.1,0.0], pca cleaned, pca minmax, e = 0.0059

# 2) all 'minmax',  Neurons= [256,256,256,256], 'relu', lr = 0.01, decay = 0.1, PcaX = 25, PcaY = 25, 
#              normalised, reg 0.00001, drps =[0.0,0.1,0.1,0.1,0.1,0.0], pca cleaned, pca minmax, e = 0.0090

# 3) all 'minmax',  Neurons= [256,256,256,256], 'relu', lr = 0.01, decay = 0.1, PcaX = 39, PcaY = 59, 
#              normalised, reg 0.00001, drps =[0.0,0.1,0.1,0.1,0.1,0.0], pca cleaned, pca minmax, e = 0.00120

# 4) all 'minmax',  Neurons= [4*128], 'sigmoid', lr = 0.001, decay = 0.1, PcaX = 39, PcaY = 59, 
#              normalised, reg 0.00, drps =[0.0,0.1,0.1,0.1,0.1,0.0], pca cleaned, pca minmax, e = 0.00120

# 5) all 'minmax',  Neurons= [4*128], 'relu', lr = 0.01, decay = 0.1, PcaX = 39, PcaY = 59, 
#              normalised, reg 0.00, drps =6*[0.0], pca cleaned, pca minmax, e = 0.0037

# 6) all 'minmax',  Neurons= [4*128], 'sigmoid', lr = 0.01, decay = 1.0, PcaX = 25, PcaY = 25, 
#              normalised, reg 0.00, drps =6*[0.0], pca cleaned, pca minmax, e = 0.00120

# 7) all 'minmax',  Neurons= [4*128], 'relu', lr = 0.01, decay = 0.1, PcaX = 39, PcaY = 59, 
#              normalised, reg 0.00001, drps =6*[0.0], pca cleaned, pca minmax, e = 0.00120



# 12 ==> big2
# 13 ==> big5

Neurons= 4*[128]
drps = [0.0,0.0,0.0,0.0,0.0,0.0]
lr2 = 0.00001
model = mytf.DNNmodel(Nin, Nout, Neurons, actLabel = ['relu','relu','relu'], drps = drps, lambReg = lr2  )

history = mytf.my_train_model( model, X_r, Y_r, num_parameters, EPOCHS, lr = 0.01, decay = 0.1, w_l = 1.0, w_mu = 0.0)
    
mytf.plot_history( history)

with open('prob2_history_7' + Run_id + '.dat', 'wb') as f:
    pickle.dump(history.history, f)
    
model.save_weights('prob2_weights_7' + Run_id)

end = timer()
print(end - start) 
