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
import elasticity_utils as elut
import pde_activation_elasticity_3param as pa
import myLibRB as rb
import myTensorflow as mytf

models_tf = []
models_pde = []

ns_trainFEA = 1000

# histories_tf_mean = {'loss':np.zeros(epochs), 'val_loss':np.zeros(epochs), 'mae_mu': np.zeros(epochs),'val_mae_mu':np.zeros(epochs), 'mae_loc': np.zeros(epochs),'val_mae_loc':np.zeros(epochs) }
# histories_pde_mean = {'loss':np.zeros(epochs), 'val_loss':np.zeros(epochs), 'mae_mu': np.zeros(epochs),'val_mae_mu':np.zeros(epochs), 'mae_loc': np.zeros(epochs),'val_mae_loc':np.zeros(epochs) }


folder2 = 'rb_bar_3param_negativePoisson/'


folder = folder2 + 'saves/'
folder3 = 'saves_test/'

num_parameters = 3
N_in = 40
N_out = 40 + num_parameters

network_width = "modified_felipe2"

# ======== building test======================
h = 0.5
L = 5.0
eps  = 0.01
xa = 2.0

box1 = elut.Box(eps,xa-eps,-h-eps,-h+eps)
box2 = elut.Box(eps,xa-eps,h-eps,h+eps)
box3 = elut.Box(xa-eps,L-eps,-h-eps,-h+eps)
box4 = elut.Box(xa-eps,L-eps,h-eps,h+eps)
box5 = elut.Box(L-eps,L+eps,-h-eps,h+eps)

regionIn = elut.Region([box3,box4])
regionOut = elut.Region([box1,box2, box5])

nAff_A = 2
nAff_f = 2

dataEx = elut.OfflineData(folder2, nAff_A, nAff_f, isTest = True)

# hacking with values of training
dataEx.param_min = np.array([-69.59540457,   3.41002456,  -0.15707422])
dataEx.param_max = np.array([ 9.26256806, 73.46698829,  0.15707537])

dataTest = elut.DataGenerator(dataEx)

dataTest.buildFeatures(['u'], regionIn, 'in', Nsample = 30 , seed = 10)
# dataTest.buildFeatures(['u', 'x+u'], regionIn, 'in2', Nsample = 30 , seed = 10)
dataTest.buildFeatures(['u','mu'], regionOut, 'out', Nsample = 30 , seed = 11)

dataTest.normaliseFeatures('out', [60,61,62], dataEx.param_min , dataEx.param_max)
uscl = elut.scaler(elut.unscale1d, np.concatenate((dataEx.param_min.reshape((3,1)), dataEx.param_max.reshape(3,1)), axis = 1))

num_parameters = dataEx.num_parameters

# Building pde model
my_pde_activation = pa.PDE_activation(dataEx.rb , dataTest.data['out'].shape[1], dataTest.dofs_loc['out'], dataEx.num_parameters, 
                                    dataEx.param_min, dataEx.param_max, [mytf.tf.identity, mytf.tf.identity, mytf.tf.cos, mytf.tf.sin], [0, 1, 2, 2] )

model_pde = mytf.PDEDNNmodel(dataEx.num_parameters, my_pde_activation.pde_mu_solver, dataTest.data['in'].shape[1], dataTest.data['out'].shape[1])

model_pde.load_weights(folder + 'weights_pde2')

# # Building tf model
model_tf = mytf.construct_dnn_model(dataTest.data['in'].shape[1], dataTest.data['out'].shape[1], network_width, act = 'linear' )
model_tf.load_weights(folder + 'weights_dnn2')


# Computing errror predictions

y_pred_pde = model_pde.predict(dataTest.data['in'])

# we need to scale input and output to compute the prediction of tf model

y_pred_tf = model_tf.predict(dataTest.data['in'])

y_test = dataTest.data['out']

# y_test[:,-num_parameters:] = uscl(y_test[:,-num_parameters:])
# y_test[:,-num_parameters:-1] = elut.convertParam2(y_test[:,-num_parameters:-1], elut.composition(elut.lame2youngPoisson,elut.lameStar2lame) )

# y_pred_pde[:,-num_parameters:] = uscl(y_pred_pde[:,-num_parameters:])
# y_pred_pde[:,-num_parameters:-1] = elut.convertParam2(y_pred_pde[:,-num_parameters:-1], elut.composition(elut.lame2youngPoisson,elut.lameStar2lame) )

# y_pred_tf[:,-num_parameters:] = uscl(y_pred_tf[:,-num_parameters:])
# y_pred_tf[:,-num_parameters:-1] = elut.convertParam2(y_pred_tf[:,-num_parameters:-1], elut.composition(elut.lame2youngPoisson,elut.lameStar2lame) )

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
    plt.scatter(y_test[:,i],y_pred_pde[:,i], marker = '+')
    xy = np.linspace(np.min(y_test[:,i]),np.max(y_test[:,i]),10)
    plt.plot(xy,xy,'-',color = 'black')
    plt.xlabel('test loc ' + str(i))
    plt.ylabel('prediction loc ' + str(i))
    plt.grid()

plt.subplots_adjust(wspace=0.3, hspace=0.3)
# plt.savefig(folder + 'scatter_loc_pde_7-12.png')
plt.show()

plt.figure(2,(13,12))
# label = ['alpha','mu','lambda']
label = ['alpha (rad)','Young','Poisson']
for i in range(1,4): 
    plt.subplot('33' + str(i))
    plt.scatter(y_pred_pde[:,-i],y_test[:,-i], marker = '+', linewidths = 0.1)
    xy = np.linspace(np.min(y_test[:,-i]),np.max(y_test[:,-i]),10)
    plt.plot(xy,xy,'-',color = 'black')
    plt.xlabel('test ' + label[i-1])
    plt.ylabel('prediction ' + label[i-1])
    plt.grid()
    
for i in range(1,4): 
    plt.subplot('33' + str(i+3))
    plt.scatter(y_test[:,-i],y_test[:,-i] - y_pred_pde[:,-i])
    plt.xlabel('test ' + label[i-1])
    plt.ylabel('error (test - pred) ' + label[i-1])
    plt.grid()

    
for i in range(1,4): 
    plt.subplot('33' + str(i+6))
    plt.scatter(y_test[:,-i],(y_test[:,-i] - y_pred_pde[:,-i])/(y_test[:,-i] + 1.0))
    plt.xlabel('test ' + label[i-1])
    plt.ylabel('error rel (test - pred)/(test + 1.0) ' + label[i-1])
    plt.grid()

plt.subplots_adjust(wspace=0.3, hspace=0.25)
# plt.savefig(folder + 'scatter_mu_u_dnn.png')
plt.show()


