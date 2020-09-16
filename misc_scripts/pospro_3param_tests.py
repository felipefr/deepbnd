#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 08:42:59 2020

@author: felipefr
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 07:30:57 2020

@author: felipefr
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import elliptic_tensorflow_routines as etr
import sys
sys.path.insert(0, '..')
import pyorb_core.rb_library.affine_decomposition as ad
import pyorb_core.rb_library.rb_manager as rm
import pde_activation_elasticity_3param as pa
from elasticity_utils import * 

models_tf = []
models_pde = []

ns_trainFEA = 1000

# histories_tf_mean = {'loss':np.zeros(epochs), 'val_loss':np.zeros(epochs), 'mae_mu': np.zeros(epochs),'val_mae_mu':np.zeros(epochs), 'mae_loc': np.zeros(epochs),'val_mae_loc':np.zeros(epochs) }
# histories_pde_mean = {'loss':np.zeros(epochs), 'val_loss':np.zeros(epochs), 'mae_mu': np.zeros(epochs),'val_mae_mu':np.zeros(epochs), 'mae_loc': np.zeros(epochs),'val_mae_loc':np.zeros(epochs) }

folder = 'saves_3param_clean/'
folder3 = 'saves_test/'

folder2 = 'rb_bar_3param/'

num_parameters = 3
N_in = 40
N_out = 40 + num_parameters

network_width = "modified_felipe"

# ======== building test======================
admNodesIn = np.loadtxt(folder2 + 'admissibleNodes_in.txt').astype('int')
admNodesOut = np.loadtxt(folder2 + 'admissibleNodes_out.txt').astype('int')
paramTest = np.loadtxt(folder2 + 'paramTest.txt')
my_snapshotsTest = np.loadtxt(folder2 + 'snapshotsTest.txt')

ns = 100 
np.random.seed(3)
noise_u = 0.0

N_in_nodes = 20 
N_out_nodes = 20 

# generation of training
X_test = np.zeros((ns, 2*N_in_nodes))
y_test = np.zeros((ns, 2*N_out_nodes + num_parameters))

y_test[:,-num_parameters:] = paramTest

my_fem_output_coordinates = np.zeros(40,dtype = 'int')

for i in range(N_in_nodes):
    rind = np.random.randint(len(admNodesIn))
    j = admNodesIn[rind]
    noise = noise_u*(np.random.rand(1)-0.5)
    X_test[:,2*i] = (1.0 + noise)*my_snapshotsTest[2*j,:] 
    noise = noise_u*(np.random.rand(1)-0.5)
    X_test[:,2*i + 1] = (1.0 + noise)*my_snapshotsTest[2*j + 1,:]

for i in range(N_out_nodes):
    rind = np.random.randint(len(admNodesOut))
    j = admNodesIn[rind]
    noise = noise_u*(np.random.rand(1)-0.5)
    y_test[:,2*i] = (1.0 + noise)*my_snapshotsTest[2*j,:]
    noise = noise_u*(np.random.rand(1)-0.5)
    y_test[:,2*i + 1] = (1.0 + noise)*my_snapshotsTest[2*j + 1,:]
    
    my_fem_output_coordinates[2*i + 1] = 2*j + 1
    my_fem_output_coordinates[2*i ] = 2*j


minmaxpar = np.array([[ 8.44978701, 23.28834597],
                     [14.3254441 , 20.72575095],
                     [-0.62524015 , 0.62798303]])


minpar = minmaxpar[:,0]
maxpar = minmaxpar[:,1]

uRef1, uRef2 = 1.1906309261856491, 0.7721084885730449

scl = scaler(scale1d,minmaxpar)
uscl = scaler(unscale1d, minmaxpar)

y_test[:,-num_parameters:] = scl(y_test[:,-num_parameters:])

print(np.min(y_test[:,-2]), np.max(y_test[:,-2]))

## ===============   Initializing Reduce basis manager =======================
base_offline_folder = './rb_bar_3param/'

num_affine_components_A = 2
num_affine_components_f = 2

# ### defining the affine decomposition structure
my_affine_decomposition = ad.AffineDecompositionHandler( )
my_affine_decomposition.set_Q( num_affine_components_A, num_affine_components_f )    # number of affine terms
my_affine_decomposition.import_rb_affine_matrices( base_offline_folder + 'ANq' )
my_affine_decomposition.import_rb_affine_vectors(  base_offline_folder + 'fNq' )

# ### building the RB manager
my_rb_manager = rm.RbManager(_affine_decomposition = my_affine_decomposition, _fom_problem = None)  

my_rb_manager.M_N = len(my_affine_decomposition.M_rbAffineFq[0]) 
print(my_rb_manager.M_N)
my_rb_manager.M_basis = np.loadtxt(folder2 + 'U.txt')[:,0:my_rb_manager.M_N]

# ===================================================================================


# Building pde model
my_pde_activation = pa.PDE_activation( my_rb_manager, N_out, my_fem_output_coordinates, num_parameters, \
                        minpar, maxpar, [tf.identity, tf.identity, tf.cos, tf.sin], [0, 1, 2, 2] )

model_pde= etr.build_pde_tensorflow_model_felipe_justModel( N_in, N_out, my_pde_activation, num_parameters, weight_mu = 0.0, \
                                    network_width="modified_felipe", optimizer='adam' , lr = 0.001)
 
model_pde.load_weights(folder + 'weights_pde_1')

# Building tf model
model_tf = etr.build_tensorflow_model_weights_justModel(N_in, N_out, num_parameters, network_width, lr = 1.e-4, decay = 1.e-2, w_l = 1.0, w_mu = 0.0)
model_tf.load_weights(folder3 + 'weights_1')


# Computing errror predictions

y_pred_pde = model_pde.predict(X_test)

# we need to scale input and output to compute the prediction of tf model
X_test = X_test/uRef1
y_test[:,:-num_parameters] = y_test[:,:-num_parameters]/uRef2

y_pred_tf = model_tf.predict(X_test)

# rescaling back
X_test = X_test*uRef1
y_test[:,:-num_parameters] = y_test[:,:-num_parameters]*uRef2
y_pred_tf[:,:-num_parameters] = y_pred_tf[:,:-num_parameters]*uRef2


# y_test[:,-num_parameters:] = uscl(y_test[:,-num_parameters:])
# y_test[:,-num_parameters:-1] = convertParam2(y_test[:,-num_parameters:-1], composition(lame2youngPoisson,lameStar2lame) )

# y_pred_pde[:,-num_parameters:] = uscl(y_pred_pde[:,-num_parameters:])
# y_pred_pde[:,-num_parameters:-1] = convertParam2(y_pred_pde[:,-num_parameters:-1], composition(lame2youngPoisson,lameStar2lame) )

# y_pred_tf[:,-num_parameters:] = uscl(y_pred_tf[:,-num_parameters:])
# y_pred_tf[:,-num_parameters:-1] = convertParam2(y_pred_tf[:,-num_parameters:-1], composition(lame2youngPoisson,lameStar2lame) )

errors_tf = y_pred_tf - y_test
errors_pde = y_pred_pde - y_test

mse = lambda x: np.mean( np.linalg.norm(x,axis=1)**2)


error_loc_tf = mse(errors_tf[:, :-num_parameters ])
error_mu_tf = mse(errors_tf[:, -num_parameters: ])

error_loc_pde = mse(errors_pde[:, :-num_parameters ])
error_mu_pde = mse(errors_pde[:, -num_parameters: ])


print('errors tf', error_loc_tf, error_mu_tf)
print('errors pde', error_loc_pde, error_mu_pde)

plt.figure(1,(12,7))
for i in range(6,12): 
    plt.subplot('23' + str(i-5))
    plt.scatter(y_test[:,i],y_pred_tf[:,i], marker = '+')
    xy = np.linspace(np.min(y_test[:,i]),np.max(y_test[:,i]),10)
    plt.plot(xy,xy,'-',color = 'black')
    plt.xlabel('test loc ' + str(i))
    plt.ylabel('prediction loc ' + str(i))
    plt.grid()

plt.subplots_adjust(wspace=0.3, hspace=0.3)
# plt.savefig(folder + 'scatter_loc_tf_7-12.png')
plt.show()

plt.figure(2,(12,8))
label = ['alpha','mu','lambda']
# label = ['alpha (rad)','Young','Poisson']
for i in range(1,4): 
    plt.subplot('23' + str(i))
    plt.scatter(y_pred_tf[:,-i],y_test[:,-i], marker = '+', linewidths = 0.1)
    xy = np.linspace(np.min(y_test[:,-i]),np.max(y_test[:,-i]),10)
    plt.plot(xy,xy,'-',color = 'black')
    plt.xlabel('test ' + label[i-1])
    plt.ylabel('prediction ' + label[i-1])
    plt.grid()
    
# for i in range(1,2): 
#     plt.subplot('23' + str(i+3))
#     plt.scatter(y_test[:,-i],y_test[:,-i] - y_pred_tf[:,-i])
#     plt.xlabel('test ' + label[i-1])
#     plt.ylabel('error (test - pred) ' + label[i-1])
#     plt.grid()

    
for i in range(1,4): 
    plt.subplot('23' + str(i+3))
    plt.scatter(y_test[:,-i],(y_test[:,-i] - y_pred_tf[:,-i])/(y_test[:,-i] + 0.5))
    plt.xlabel('test ' + label[i-1])
    plt.ylabel('error rel (test - pred)/(test + 0.5) ' + label[i-1])
    plt.grid()

plt.subplots_adjust(wspace=0.3, hspace=0.25)
# plt.savefig(folder + 'scatter_mu.png')
plt.show()


