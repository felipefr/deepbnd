from __future__ import absolute_import, division, print_function
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt
import numpy as np
import myLibRB as rb

import myTensorflow as mytf
import pickle
from sklearn.preprocessing import MinMaxScaler
import random
import pde_activation_elasticity_3param as pa

from timeit import default_timer as timer

import elasticity_utils as elut

folder = 'rb_bar_3param_negativePoisson/'

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

dataEx = elut.OfflineData(folder, nAff_A, nAff_f)

dataTrain = elut.TrainingDataGenerator(dataEx)

dataTrain.buildFeatures(['u','x+u'], regionIn, 'in', Nsample = 30 , seed = 10)
dataTrain.buildFeatures(['u','mu'], regionOut, 'out', Nsample = 30 , seed = 11)

dataTrain.normaliseFeatures('out', [60,61,62], dataEx.param_min , dataEx.param_max)

mytf.tf.set_random_seed(2)

Run_id = ''
EPOCHS = 500

weight_mu = 0.0

chosen_network_width = 'modified_felipe2'

runs = 3


print( "######################################################################## ")
print( "############### TENSORFLOW + PDEs mu_{in} -> mu_{pde} ################## ")
print( "######################################################################## ")

start = timer()

typeOfModel = 'pdednn'

for i in range(2,runs):
    Run_id = str(1 + i)

    if(typeOfModel == 'pdednn'):
        w_mu = 0.0
        w_l = 1.0
        decay = 0.1
        lr = 0.0001
        my_pde_activation = pa.PDE_activation(dataEx.rb , dataTrain.data['out'].shape[1], dataTrain.dofs_loc['out'], dataEx.num_parameters, 
                                            dataEx.param_min, dataEx.param_max, [mytf.tf.identity, mytf.tf.identity, mytf.tf.cos, mytf.tf.sin], [0, 1, 2, 2] )
    

        model = mytf.PDEDNNmodel(dataEx.num_parameters, my_pde_activation.pde_mu_solver, dataTrain.data['in'].shape[1], dataTrain.data['out'].shape[1])
    
    elif(typeOfModel == 'dnn'):
        w_mu = 0.01
        w_l = 1.0
        decay = 0.01
        lr = 0.001
        model = mytf.construct_dnn_model(dataTrain.data['in'].shape[1], dataTrain.data['out'].shape[1], chosen_network_width, act = 'linear' )
        
    history = mytf.my_train_model( model, dataTrain.data['in'], dataTrain.data['out'], dataEx.num_parameters, EPOCHS, lr = lr, decay = decay, w_l = w_l, w_mu = w_mu)
    
        
    mytf.plot_history( history)
    
    with open(folder + 'saves/history_' + typeOfModel + Run_id + '.dat', 'wb') as f:
        pickle.dump(history.history, f)
        
    
    # pde_model.save('./saves_3param/Model_64_pde_wMu' + Run_id + '.hd5')

    model.save_weights(folder + 'saves/weights_' + typeOfModel + Run_id)


end = timer()
print(end - start) # Time in seconds, e.g. 5.38091952400282