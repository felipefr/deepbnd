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




# filesToMerge = [ folderData + 'prob1_{0}dataset.hdf5'.format(i) for i in range(5)]
# myHDF5.merge(filesToMerge, 'prob1_dataset.hdf5', ['Unique/X'], ['Unique/X'], mode = 'w')
# myHDF5.merge(filesToMerge, 'prob1_dataset.hdf5', ['Unique/Y'], ['Unique/Y'], mode = 'r+')

ns = 10000

with h5py.File('prob1_dataset.hdf5', 'r') as f:
    X_train = np.array(f['Unique/X'][:ns,:]) # displacements
    Y_train = np.array(f['Unique/Y'][:ns,:]) # stesses



# svdY = TruncatedSVD(n_components=20, algorithm = 'arpack')

subsets = {'stress' : np.arange(0,60,3) , 'x' : np.sort(np.concatenate((np.arange(1,60,3), np.arange(2,60,3))))}
ind = dman.getPCAcut(Y_train, subsets, xr0min=-3.0, xr0max=0.0, xr1min=-2.1, xr1max=2.5)    

subsets3 = {'stress' : np.arange(0,60,3) , 'x' : np.sort(np.concatenate((np.arange(1,60,3), np.arange(2,60,3))))}
Y_train = Y_train[ind,:]
Y_r, TY = dman.getCombinedPCAnormalisation(Y_train, subsets3,  N=25)

subsets2 = {'all' : []}
X_train = X_train[ind,:]
X_r, TX = dman.getCombinedPCAnormalisation(X_train, subsets2, N=25)

X_rr = dman.getCombinedPCAreconstruction(X_r,TX)
Y_rr = dman.getCombinedPCAreconstruction(Y_r,TY)

print(np.linalg.norm(X_train - X_rr)/60000)
print(np.linalg.norm(Y_rr - Y_train)/60000)


Nin = X_r.shape[1]
Nout = Y_r.shape[1]

mytf.tf.set_random_seed(3)

Run_id = ''
EPOCHS = 500

weight_mu = 0.0

num_parameters = 0 # parameters that are taking into account into the model for minimization.

start = timer()

# good arch
# 1) Neurons= [20,20,20], 'tanh', lr = 0.05, decay = 0.5, PcaX = 10, PcaY = 10, unnormalised, no reg, no drop, e = 0.169
# 2) Neurons= [10,10,10], 'tanh', lr = 0.1, decay = 0.1, PcaX = 10, PcaY = 10, unnormalised, no reg, no drop, e = 0.135
# 3) Neurons= [20,20,20], 'linear', lr = 0.001, decay = 0.1, PcaX = 10, PcaY = 10, normalised, no reg, no drop, pca cleaned, e = 0.1337
# 4) Neurons= [64,64,64], 'linear', lr = 0.001, decay = 0.1, PcaX = 10, PcaY = 10, normalised, no reg, no drop, pca cleaned, e = 0.1044
# 5) Neurons= [64,64,64], 'linear', lr = 0.001, decay = 0.1, PcaX = 10, PcaY = 10, 
#              normalised, no reg, no drop, pca cleaned, pca minmax, e = 0.0093 , but bad generalization
# 5) Neurons= [64,64,64], 'relu', lr = 0.01, decay = 0.1, PcaX = 10, PcaY = 10, 
#              normalised, no reg, no drop, pca cleaned, pca minmax, e = 0.0087 , but bad generalization
# 6) Neurons= [64,64,64], 'relu', lr = 0.01, decay = 0.1, PcaX = 10, PcaY = 10, 
#              normalised, no reg, drps =[0.2,0.2,0.2,0.2,0.2], pca cleaned, pca minmax, e = 0.0150 , very good gereralization
# 7) Neurons= [64,64,64,64], 'relu', lr = 0.01, decay = 0.1, PcaX = 10, PcaY = 10, 
#              normalised, no reg, drps =[0.2,0.2,0.2,0.2,0.2, 0.2], pca cleaned, pca minmax, e = 0.0152 , very good gereralization
# 8) Neurons= [64,64,64,64], 'relu', lr = 0.01, decay = 0.1, PcaX = 10, PcaY = 10, 
#              normalised, no reg, drps =[0.2,0.2,0.2,0.2,0.2, 0.2], pca cleaned, pca minmax, e = 0.0154 , very good gereralization, corrected X_train_t

# 9) Neurons= [256,256,256,256], 'relu', lr = 0.001, decay = 0.1, PcaX = 10, PcaY = 10, 
#              normalised, no reg, drps =[0.2,0.3,0.3,0.3,0.3,0.0], pca cleaned, pca minmax, e = 0.0097 , very good gereralization, corrected X_train_t

# 10) Neurons= [256,256,256,256], 'relu', lr = 0.001, decay = 0.1, PcaX = 25, PcaY = 25, 
#              normalised, reg 0.0001, drps =[0.2,0.3,0.3,0.3,0.3,0.0], pca cleaned, pca minmax, e = 0.0062 , very good gereralization, corrected X_train_t

# 11) Neurons= [256,256,256,256], 'relu', lr = 0.001, decay = 0.1, PcaX = 5, PcaY = 5, 
#              normalised, reg 0.0001, drps =[0.2,0.3,0.3,0.3,0.3,0.0], pca cleaned, pca minmax, e = 0.0158 , very good gereralization, corrected X_train_t

# 12) Neurons= [256,256,256,256], 'relu', lr = 0.0001, decay = 0.1, PcaX = 25, PcaY = 25, 
#              normalised, reg 0.00001, drps =[0.0,0.1,0.1,0.1,0.1,0.0], pca cleaned, pca minmax, e = 0.003 , very good gereralization, corrected

# 13) all 'minmax',  Neurons= [256,256,256,256], 'relu', lr = 0.0001, decay = 0.1, PcaX = 25, PcaY = 25, 
#              normalised, reg 0.00001, drps =[0.0,0.1,0.1,0.1,0.1,0.0], pca cleaned, pca minmax, e = 0.003 , very good gereralization, corrected


# 12 ==> big2
# 13 ==> big5

Neurons= [256,256,256,256]
drps =[0.0,0.1,0.1,0.1,0.1,0.0]
lr2 = 0.00001
model = mytf.DNNmodel(Nin, Nout, Neurons, actLabel = ['relu','relu','relu'], drps = drps, lambReg = lr2  )

history = mytf.my_train_model( model, X_r, Y_r, num_parameters, EPOCHS, lr = 0.001, decay = 0.1, w_l = 1.0, w_mu = 0.0)
    
mytf.plot_history( history)

with open('prob1_history_big5' + Run_id + '.dat', 'wb') as f:
    pickle.dump(history.history, f)
    
model.save_weights('prob1_weights_big5' + Run_id)

end = timer()
print(end - start) 
