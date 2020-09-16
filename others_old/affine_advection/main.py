from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
print(tf.__version__)

tf.enable_eager_execution( )

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../../../pyorb/')
sys.path.insert(0, '../../../pyorb/examples/affine_advection_cpp')

import pyorb_core.rb_library.rb_manager as rm
import pyorb_core.rb_library.affine_decomposition as ad
import pyorb_core.pde_problem.parameter_handler as ph
import pickle


#########################################################
################ SETTING CPP ENGINE ##################
#########################################################

# cpp_library_path = '/fakefolder/fakelibrary.so'

# playing around with engine manager 
# my_cpp_engine_manager = mee.external_engine_manager( 'cpp', cpp_library_path )
# my_cpp_engine_manager.start_engine( )
my_cpp_external_engine = 0 # Felipe : This is wrong, but it just to initiliaze other instances

#########################################################
################# RB & PROBLEM SETTING ##################
#########################################################

# Parameter ranges
mu0_min = 0.5; mu0_max = 10.
mu1_min = 0; mu1_max = 30.

ns_train = 350
ns_test  = 150

param_min = np.array([mu0_min, mu1_min])
param_max = np.array([mu0_max, mu1_max])
num_parameters = param_min.shape[0]

# preparing the parameter handler
my_parameter_handler = ph.Parameter_handler( )
my_parameter_handler.assign_parameters_bounds( param_min, param_max )

# define the fem problem
import affine_advection as aap

my_aap = aap.affine_advection_problem( my_parameter_handler )

fom_specifics = {
        'model'             : 'affine_advection', \
        'datafile_path'     : './simulation_data/data'}

my_aap.configure_fom( my_cpp_external_engine, fom_specifics )

base_offline_folder = './offline_affine_advection/'

print( 'The base offline folder is %s ' % base_offline_folder )

num_affine_components_A = 3
num_affine_components_f = 3

# defining the affine decomposition structure
my_affine_decomposition = ad.AffineDecompositionHandler( )
my_affine_decomposition.set_Q( num_affine_components_A, num_affine_components_f )    # number of affine terms

# building the RB manager
my_rb_manager = rm.RbManager( my_affine_decomposition, my_aap )

snapshots_file = base_offline_folder + 'snapshots_affine_advection_lifting.txt'
my_rb_manager.import_snapshots_matrix( snapshots_file )
my_rb_manager.import_snapshots_parameters( base_offline_folder + 'offline_parameters.data' )

my_rb_manager.build_rb_approximation( ns_train, 10**(-6) ) # Felipe 10-6 gives 28 and 10-7 gives 38, I needed 28 because is already computed. 

my_affine_decomposition.import_rb_affine_matrices( base_offline_folder + 'rb_affine_components_affine_advection_lifting_A' ) # Felipe: Order changed with build_rb_approximation, which resets it
my_affine_decomposition.import_rb_affine_vectors(  base_offline_folder + 'rb_affine_components_affine_advection_lifting_f' ) # Felipe: Order changed with build_rb_approximation, which resets it
 
# my_rb_manager.test_rb_solver( 1 ) # Felipe: skipped because to test it needs access to the solver

num_theta_functions = num_affine_components_A
train_parameters = my_rb_manager.get_offline_parameters( )

my_rb_manager.import_test_parameters( base_offline_folder + 'test_parameters.data' )
my_rb_manager.import_test_snapshots_matrix( base_offline_folder + 'snapshots_test.txt' )
test_parameters = my_rb_manager.get_test_parameter_matrix( )

#%%

import generate_data as gd

########################################################################
#################### TRAINING AND TEST DATA SETTING ####################
########################################################################

n_dofs = my_rb_manager.get_snapshots_matrix().shape[0]

number_of_fem_coordinates = 20

sampling = 'random'

fem_coordinates = gd.generate_fem_coordinates_from_list( number_of_fem_coordinates, "input_arrays/set0_yplane.txt", \
                                              sampling )
fem_coordinates = np.sort( fem_coordinates )

np.save("output_arrays/fem_coordinates", fem_coordinates)

number_of_output_coordinates = 20

fem_output_coordinates = gd.generate_fem_coordinates_from_list( number_of_output_coordinates, "input_arrays/set1_yplane.txt", \
                                                      sampling )

fem_output_coordinates = np.sort( fem_output_coordinates )
np.save("output_arrays/fem_output_coordinates", fem_output_coordinates)


#%%
# number of selected parameters for the training
my_rb_manager.M_get_test = False
noise_magnitude = 0.0

X_train, y_train = gd.generate_fem_training_data( ns_train, fem_coordinates, fem_output_coordinates, 
                                                  my_rb_manager, 2, \
                                                  param_min, param_max, \
                                                  my_parameter_handler, noise_magnitude,
                                                  data_file = base_offline_folder + 'offline_parameters.data' )

n_final_samples = 10000

if n_final_samples > ns_train:
    X_train, y_train = gd.expand_train_with_rb( X_train, y_train, n_final_samples, \
                                                fem_coordinates,
                                                fem_output_coordinates, \
                                                my_rb_manager, 2, \
                                                param_min, param_max, \
                                                my_aap )

X_train = X_train[:n_final_samples,:]
y_train = y_train[:n_final_samples,:]
#%%

my_rb_manager.M_get_test = True

noise_magnitude = 0.0
X_test, y_test = gd.generate_fem_training_data( ns_test, fem_coordinates, fem_output_coordinates, 
                                                  my_rb_manager, 2, \
                                                  param_min, param_max, \
                                                  my_parameter_handler, noise_magnitude,
                                                  data_file = base_offline_folder + 'test_parameters.data' )

#%%

    
runs = 10 

import elliptic_tensorflow_routines as etr
import pde_activation as pa

for i in range(runs):

######################################################################
################### TENSORFLOW mu_{in} -> mu_{pde} ###################
######################################################################

    Run_id = str(1 + i)
        
    print( "######################################################################## ")
    print( "################### TENSORFLOW mu_{in} -> mu_{pde} ################### ")
    print( "######################################################################## ")
    
    EPOCHS = 500
    chosen_network_width = 'constant'
    
    tf_model,history = etr.build_tensorflow_model( X_train, y_train, EPOCHS, network_width=chosen_network_width, lr = 0.0002 )
    etr.plot_history(history)
    
    with open('./saves_lr00002/historyModel_' + Run_id + '.dat', 'wb') as f:
        pickle.dump(history.history, f)
        
    
    tf_model.save('./saves_lr00002/model_' + Run_id + '.hd5')
    
    ########################################################################
    ################# TENSORFLOW + PDEs mu_{in} -> mu_{pde} ################
    ########################################################################
    
    print( "######################################################################## ")
    print( "############### TENSORFLOW + PDEs mu_{in} -> mu_{pde} ################## ")
    print( "######################################################################## ")
    
    EPOCHS = 500
    
    weight_mu = 0.0
    
    chosen_network_width = 'constant'
    
    my_pde_activation = pa.PDE_activation( my_rb_manager, number_of_output_coordinates + 2, fem_output_coordinates, 2, \
                           param_min, param_max, [tf.identity, tf.sin, tf.cos], [0, 1, 1] )
    
    pde_model,history = etr.build_pde_tensorflow_model( X_train, y_train, EPOCHS, my_pde_activation, 2, weight_mu, \
                                        network_width=chosen_network_width, optimizer='adam' , lr = 0.0002)
        
    etr.plot_history( history )
    
    
    with open('./saves_lr00002/historyModel_pde_' + Run_id + '.dat', 'wb') as f:
        pickle.dump(history.history, f)
        
    
    pde_model.save('./saves_lr00002/model_pde_' + Run_id + '.hd5')


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
