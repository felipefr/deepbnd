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



base_offline_folder = './rb_bar_3param/'

# in total ns = 500
ns_train = 1000
ns_test  = 100
n_final_samples = 1000

num_affine_components_A = 2
num_affine_components_f = 2
num_parameters = 3

admNodesIn = np.loadtxt(base_offline_folder + 'admissibleNodes_in_boundary.txt').astype('int')
admNodesIn_pos = np.loadtxt(base_offline_folder + 'admissible_in_boundary_positions.txt').astype('int')

admNodesOut = np.loadtxt(base_offline_folder + 'admissibleNodes_out_boundary.txt').astype('int')
admNodesOut_pos = np.loadtxt(base_offline_folder + 'admissible_out_boundary_positions.txt').astype('int')

type_in = 'x' # it may be just 'u' or just 'x'

type_out = 'u_and_mu' # it may be just 'u'

Nin_nodes = 20
Nout_nodes = 20

if type_in == 'u_and_x': 
    Nin = 4*Nin_nodes
elif type_in == 'u' or type_in == 'x' :
    Nin = 2*Nin_nodes

if type_out == 'u_and_mu':
    Nout = 2*Nout_nodes + num_parameters
elif type_out == 'u':
    Nout = 2*Nout_nodes
    
nsRB = 10000
Nh = 2*561

my_snapshotsFEA = np.zeros( (Nh,ns_train + nsRB)) 

my_snapshotsFEA[:,:ns_train] = np.loadtxt(base_offline_folder + 'snapshotsFEA.txt')
my_snapshotsFEA[:,ns_train:] = np.loadtxt(base_offline_folder + 'snapshotsRB.txt')

ns = len(my_snapshotsFEA[0])

np.random.seed(3)

# generation of training
X_train = np.zeros((ns, Nin))
y_train = np.zeros((ns, Nout))

my_fem_output_coordinates = np.zeros(2*Nout_nodes, dtype = 'int')

for i in range(Nin_nodes):
    rind = np.random.randint(len(admNodesIn))
    j = admNodesIn[rind]

    if(type_in == 'u_and_x'):
        X_train[:,4*i: 4*i+2] = my_snapshotsFEA[2*j:2*(j+1),:].transpose() 
        X_train[:,4*i + 2: 4*(i+1)] = admNodesIn_pos[rind,:]
    elif(type_in == 'u'):
        X_train[:,2*i:2*(i+1)] = my_snapshotsFEA[2*j:2*(j+1),:].transpose()  
    elif(type_in == 'x'):
        X_train[:,2*i:2*(i+1)] = admNodesIn_pos[rind,:]

for i in range(Nout_nodes):
    rind = np.random.randint(len(admNodesOut))
    j = admNodesIn[rind]
    
    y_train[:,2*i:2*(i+1)] = my_snapshotsFEA[2*j:2*(j+1),:].transpose() 
    
    my_fem_output_coordinates[2*i:2*(i+1)] = np.arange(2*j,2*(j+1))


if(type_out == 'u_and_mu'):
    paramRB = np.loadtxt(base_offline_folder + 'paramRB.txt')
    paramFEA = np.loadtxt(base_offline_folder + 'paramFEA.txt')
    y_train[:ns_train,-num_parameters:] = paramFEA
    y_train[ns_train:,-num_parameters:] = paramRB

    param_min = np.zeros(num_parameters)
    param_max = np.zeros(num_parameters)
    
    for i in range(num_parameters):
        param_min[i] = np.min(y_train[:, Nout - num_parameters + i])
        param_max[i] = np.max(y_train[:, Nout - num_parameters + i])
        y_train[:,Nout - num_parameters + i] = (y_train[:,Nout - num_parameters + i] - param_min[i])/(param_max[i] - param_min[i])

mytf.tf.set_random_seed(2)

Run_id = ''
EPOCHS = 500

weight_mu = 0.0

chosen_network_width = 'modified_felipe2'

runs = 1


print( "######################################################################## ")
print( "############### TENSORFLOW + PDEs mu_{in} -> mu_{pde} ################## ")
print( "######################################################################## ")

start = timer()

for i in range(runs):
    Run_id = str(1 + i)

    my_pde_activation = pa.PDE_activation(rb.getRBmanager(num_affine_components_A, num_affine_components_f, base_offline_folder), 
                                          Nout, my_fem_output_coordinates, num_parameters, 
                                            param_min, param_max, [mytf.tf.identity, mytf.tf.identity, mytf.tf.cos, mytf.tf.sin], [0, 1, 2, 2] )
    
    # model = mytf.construct_pdednn_model(Nin, Nout, num_parameters, my_pde_activation)
    model = mytf.PDEDNNmodel(num_parameters, my_pde_activation.pde_mu_solver, Nin, Nout)
    history = mytf.my_train_model( model, X_train, y_train, num_parameters, EPOCHS, lr = 0.001, decay = 1.e-1, w_l = 1.0, w_mu = 0.0)
    
    
    # model = mytf.construct_dnn_model(Nin, Nout, chosen_network_width, act = 'linear' )
        
    # history = mytf.my_train_model( model, X_train, y_train, num_parameters, EPOCHS, lr = 0.001, decay = 1.e-1, w_l = 1.0, w_mu = 0.01)
        
    mytf.plot_history( history)
    
    with open('./saves_3param_clean/historyModel_boundary_Nin20_just_x_pde' + Run_id + '.dat', 'wb') as f:
        pickle.dump(history.history, f)
        
    
    # pde_model.save('./saves_3param/Model_64_pde_wMu' + Run_id + '.hd5')

    model.save_weights('./saves_3param_clean/weights_boundary_Nin20_just_x_pde' + Run_id)


end = timer()
print(end - start) # Time in seconds, e.g. 5.38091952400282