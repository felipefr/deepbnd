from __future__ import absolute_import, division, print_function
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# tf.compat.v1.enable_eager_execution()

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../../../pyorb/')
# sys.path.insert(0, '../../../pyorb/examples/affine_advection_cpp')


# import pyorb_core.pde_problem.fom_problem as fm
import pyorb_core.rb_library.affine_decomposition as ad
import pyorb_core.pde_problem.parameter_handler as ph
import pyorb_core.rb_library.rb_manager as rm
import generate_data_elasticity as gd
import elliptic_tensorflow_routines as etr
import pickle
from sklearn.preprocessing import MinMaxScaler
import random

my_matlab_external_engine = 0 # dummy

#########################################################
################# RB & PROBLEM SETTING ##################
#########################################################

base_offline_folder = './rb_bar_3param/'

# in total ns = 500
ns_train = 1000
ns_test  = 100
n_final_samples = 1000

paramLimits = np.loadtxt(base_offline_folder + 'paramLimits.txt')

print(paramLimits)
param_min = paramLimits[:,0]
param_max = paramLimits[:,1]
num_parameters = param_min.shape[0]

# # preparing the parameter handler
my_parameter_handler = ph.Parameter_handler( )
my_parameter_handler.assign_parameters_bounds( param_min, param_max )

# define the fem problem
import elasticity_problem as ep

my_ep = ep.elasticity_problem( my_parameter_handler )

fom_specifics = {
        'model'             : 'elasticity', \
        'datafile_path'     : './rb_bar_lame/'}

my_ep.configure_fom( my_matlab_external_engine, fom_specifics )

num_affine_components_A = 2
num_affine_components_f = 2

# ### defining the affine decomposition structure
my_affine_decomposition = ad.AffineDecompositionHandler( )
my_affine_decomposition.set_Q( num_affine_components_A, num_affine_components_f )    # number of affine terms

# ### building the RB manager
my_rb_manager = rm.RbManager(_affine_decomposition = my_affine_decomposition, _fom_problem = my_ep) # todo: fom_problem is missed by the moment, and I hope it can missed permanently  

my_rb_manager.import_snapshots_matrix( base_offline_folder + 'snapshotsFEA.txt' )
my_rb_manager.import_snapshots_parameters( base_offline_folder + 'paramFEA.txt' )

my_rb_manager.build_rb_approximation( ns_train, 1.e-8 ) # The same used in the Rb generation 

my_affine_decomposition.import_rb_affine_matrices( base_offline_folder + 'ANq' )
my_affine_decomposition.import_rb_affine_vectors(  base_offline_folder + 'fNq' )
 
train_parameters = my_rb_manager.get_offline_parameters( )

my_rb_manager.import_test_parameters( base_offline_folder + 'paramTest.txt' )
my_rb_manager.import_test_snapshots_matrix( base_offline_folder + 'snapshotsTest.txt' )
test_parameters = my_rb_manager.get_test_parameter_matrix( )

########################################################################
#################### TRAINING AND TEST DATA SETTING ####################
########################################################################

n_dofs = my_rb_manager.get_snapshots_matrix().shape[0]

number_of_fem_coordinates = 60

sampling = 'random'

fem_coordinates = gd.generate_fem_coordinates_from_list( number_of_fem_coordinates, base_offline_folder + "admissibleDofs_in.txt", sampling )
fem_coordinates = np.sort( fem_coordinates )

np.save(base_offline_folder + "fem_coordinates", fem_coordinates)

number_of_output_coordinates = 60

fem_output_coordinates = gd.generate_fem_coordinates_from_list( number_of_output_coordinates, base_offline_folder + "admissibleDofs_out.txt", sampling )

fem_output_coordinates = np.sort( fem_output_coordinates )
np.save(base_offline_folder + "fem_output_coordinates", fem_output_coordinates)


# #%%
# # number of selected parameters for the training
my_rb_manager.M_get_test = False
noise_magnitude = 0.0

X_train, y_train = gd.generate_fem_training_data( ns_train, fem_coordinates, fem_output_coordinates, 
                                                  my_rb_manager, num_parameters, \
                                                  param_min, param_max, \
                                                  my_parameter_handler, noise_magnitude,
                                                  data_file = base_offline_folder + 'paramFEA.txt')

    
    
admNodesIn = np.loadtxt(base_offline_folder + 'admissibleNodes_in.txt').astype('int')
admNodesOut = np.loadtxt(base_offline_folder + 'admissibleNodes_out.txt').astype('int')

Nin = 20
Nout = 20

nsRB = 10000
Nh = 2*561

my_snapshotsFEA = np.zeros( (Nh,ns_train + nsRB)) 

my_snapshotsFEA[:,:ns_train] = np.loadtxt(base_offline_folder + 'snapshotsFEA.txt')
my_snapshotsFEA[:,ns_train:] = np.loadtxt(base_offline_folder + 'snapshotsRB.txt')

ns = len(my_snapshotsFEA[0])

print(ns)


np.random.seed(3)
noise_u = 0.0

# generation of training
X_train = np.zeros((ns, 2*Nin))
y_train = np.zeros((ns, 2*Nout + num_parameters))

for i in range(Nin):
    rind = np.random.randint(len(admNodesIn))
    j = admNodesIn[rind]
    print(j)
    noise = noise_u*(np.random.rand(1)-0.5)
    X_train[:,2*i] = (1.0 + noise)*my_snapshotsFEA[2*j,:] 
    noise = noise_u*(np.random.rand(1)-0.5)
    X_train[:,2*i + 1] = (1.0 + noise)*my_snapshotsFEA[2*j + 1,:]
    
print('out')

for i in range(Nout):
    rind = np.random.randint(len(admNodesOut))
    j = admNodesIn[rind]
    print(j)
    noise = noise_u*(np.random.rand(1)-0.5)
    y_train[:,2*i] = (1.0 + noise)*my_snapshotsFEA[2*j,:]
    noise = noise_u*(np.random.rand(1)-0.5)
    y_train[:,2*i + 1] = (1.0 + noise)*my_snapshotsFEA[2*j + 1,:]


paramRB = np.loadtxt(base_offline_folder + 'paramRB.txt')
y_train[:ns_train,-num_parameters:] = train_parameters
y_train[ns_train:,-num_parameters:] = paramRB

uRef1 = np.max( np.abs(X_train.flatten()))
uRef2 = np.max( np.abs(y_train[:,:-num_parameters].flatten()) )

# print(uRef1,uRef2)

# input()

X_train = X_train/uRef1
y_train[:,:-num_parameters] = y_train[:,:-num_parameters]/uRef2

for i in range(1,num_parameters+1):
    y_train[:,-i] = (y_train[:,-i] - np.min(y_train[:,-i]))/(np.max(y_train[:,-i]) - np.min(y_train[:,-i]))

# training for the trivial problem u = f(mu)
# X_train = np.zeros((ns, num_parameters))
# y_train = np.zeros((ns, 2*Nout))

# for i in range(Nout):
#     rind = random.randint(0,len(admNodesOut)-1)
#     j = admNodesIn[rind]
    
#     y_train[:,2*i] = my_snapshotsFEA[2*j,:]
#     y_train[:,2*i + 1] = my_snapshotsFEA[2*j + 1,:]

# X_train = train_parameters

# for i in range(num_parameters):
#     X_train[:,i] = (X_train[:,i] - np.min(X_train[:,i]))/(np.max(X_train[:,i]) - np.min(X_train[:,i]))


# uRef2 = np.max( np.abs(y_train.flatten()))

# y_train = y_train/uRef2

# end train for the trivial


       

# X_train = X_train[:n_final_samples,:]
# y_train = y_train[:n_final_samples,:]

# scalerX = MinMaxScaler()
# scalerY = MinMaxScaler()

# scalerX.fit(X_train)
# scalerY.fit(y_train)

# X_train_norm = scalerX.transform(X_train)
# y_train_norm = scalerY.transform(y_train)


######################################################################
################### TENSORFLOW mu_{in} -> mu_{pde} ###################
######################################################################

tf.set_random_seed(2)

runs = 10

for i in range(runs):
    Run_id = str(1 + i)
    print( "######################################################################## ")
    print( "################### TENSORFLOW mu_{in} -> mu_{pde} ################### ")
    print( "######################################################################## ")

    EPOCHS = 500
    chosen_network_width = 'modified_felipe'

    tf_model,history = etr.build_tensorflow_model_weights( X_train , y_train, 3, EPOCHS, network_width=chosen_network_width, 
                                                          lr = 1.0e-4, decay = 1.0e-3, w_l = 0.0, w_mu = 1.0)
    etr.plot_history(history)

    with open('./saves_3param_clean/historyModel_' + Run_id + '.dat', 'wb') as f:
        pickle.dump(history.history, f)
        
    tf_model.save_weights('./saves_3param_clean/weights_' + Run_id)
#%%


# input()


########################################################################
################# TENSORFLOW + PDEs mu_{in} -> mu_{pde} ################
########################################################################

# import pde_activation_elasticity_3param as pa


# print( "######################################################################## ")
# print( "############### TENSORFLOW + PDEs mu_{in} -> mu_{pde} ################## ")
# print( "######################################################################## ")

# Run_id = ''
# EPOCHS = 500

# weight_mu = 0.0

# chosen_network_width = 'constant'

# my_pde_activation = pa.PDE_activation( my_rb_manager, number_of_output_coordinates + num_parameters, fem_output_coordinates, num_parameters, \
#                         param_min, param_max, [tf.identity, tf.identity, tf.cos, tf.sin], [0, 1, 2, 2] )

# pde_model,history = etr.build_pde_tensorflow_model( X_train, y_train, EPOCHS, my_pde_activation, num_parameters, weight_mu, \
#                                     network_width=chosen_network_width, optimizer='adam' , lr = 0.0001)
    
# etr.plot_history( history, './saves_pde/loss_ns10000_8')

# with open('./saves_pde/historyModel_' + Run_id + '.dat', 'wb') as f:
#     pickle.dump(history.history, f)
    

# pde_model.save('./saves_pde/' + Run_id + '.hd5')

# #%%

# [loss, mae] = pde_model.evaluate(X_test, y_test, verbose=0)
# print("\n\nPDE Loss and mae are %f   %f" % (loss, mae))

# [tf_loss, tf_mae] = tf_model.evaluate(X_test, y_test, verbose=0)
# print("\n\nTF  Loss and mae for are %f   %f" % (tf_loss, tf_mae))

# #%%

# import elliptic_tensorflow_routines as etr

# folder = 'results_parameter_identification'

# pde_error_param, pde_error_test = etr.evaluate_model( pde_model, 'PDE', 
#                                                       X_test, y_test, 2 )

# tf_error_test_param, tf_error_test = etr.evaluate_model( tf_model, 'TF', 
#                                                          X_test, y_test, 2 )

# for iQ in range( 2 ):
#     if tf_error_test_param[iQ] > pde_error_param[iQ]:
#         print( "PDE wins for Theta_" + str(iQ) + "( mu )    !     PDE: %f vs TF: %f " %( pde_error_param[iQ], tf_error_test_param[iQ] ) )
#     else:
#         print( "TF  wins for Theta_" + str(iQ) + "( mu )    !     PDE: %f vs TF: %f " %( pde_error_param[iQ], tf_error_test_param[iQ] ) )


# if tf_error_test > pde_error_test:
#     print( "PDE wins in general   !     PDE: %f vs TF: %f " %( pde_error_test, tf_error_test ) )
# else:
#     print( "TF  wins in general   !     PDE: %f vs TF: %f " %( pde_error_test, tf_error_test ) )
    

# #%%
    
# index_to_export = 10

# res = pde_model.predict( X_test )

# np.save('output_arrays/res', res[index_to_export,:])
# np.save('output_arrays/y_test', y_test[index_to_export,:])
