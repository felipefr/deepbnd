from __future__ import absolute_import, division, print_function
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


tf.compat.v1.enable_eager_execution()

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../../../pyorb/')
# sys.path.insert(0, '../../../pyorb/examples/affine_advection_cpp')


# import pyorb_core.pde_problem.fom_problem as fm
import pyorb_core.rb_library.affine_decomposition as ad
import pyorb_core.pde_problem.parameter_handler as ph
import pyorb_core.rb_library.rb_manager as rm
import pyorb_core.tpl_managers.external_engine_manager as mee
import generate_data_elasticity as gd

#########################################################
################ SETTING CPP ENGINE ##################
#########################################################
matlab_library_path = '/Users/felipefr/Dropbox/RB_NN/feamat/'
matlab_pyorb_interface = '/Users/felipefr/Dropbox/RB_NN/pyorb-matlab-api/' # Todo: still not used but should
my_matlab_engine_manager = mee.external_engine_manager( 'matlab', matlab_library_path)
my_matlab_engine_manager.start_engine( )
my_matlab_external_engine = my_matlab_engine_manager.get_external_engine( )

#########################################################
################# RB & PROBLEM SETTING ##################
#########################################################

# Parameter ranges
nu_min = 0.1;  nu_max = 0.40
E_min = 100.0; E_max = 150.0

# in total ns = 500
ns_train = 500
ns_test  = 100

param_min = np.array([nu_min, E_min])
param_max = np.array([nu_max, E_max])
num_parameters = param_min.shape[0]

# # preparing the parameter handler
my_parameter_handler = ph.Parameter_handler( )
my_parameter_handler.assign_parameters_bounds( param_min, param_max )

# define the fem problem
import elasticity_problem as ep

my_ep = ep.elasticity_problem( my_parameter_handler )

fom_specifics = {
        'model'             : 'elasticity', \
        'datafile_path'     : './offline/'}

my_ep.configure_fom( my_matlab_external_engine, fom_specifics )

base_offline_folder = './offline/'

# print( 'The base offline folder is %s ' % base_offline_folder )

num_affine_components_A = 2
num_affine_components_f = 1

# ### defining the affine decomposition structure
my_affine_decomposition = ad.AffineDecompositionHandler( )
my_affine_decomposition.set_Q( num_affine_components_A, num_affine_components_f )    # number of affine terms

# ### building the RB manager
my_rb_manager = rm.RbManager(_affine_decomposition = my_affine_decomposition, _fom_problem = my_ep) # todo: fom_problem is missed by the moment, and I hope it can missed permanently  

my_rb_manager.import_snapshots_matrix( base_offline_folder + 'snapshots.txt' )
my_rb_manager.import_snapshots_parameters( base_offline_folder + 'param.txt' )

my_rb_manager.build_rb_approximation( ns_train, 0.0004 ) # Felipe This tolerance value gives N = 20, which is the avalaible dimension for ANq  

my_affine_decomposition.import_rb_affine_matrices( base_offline_folder + 'ANq' )
my_affine_decomposition.import_rb_affine_vectors(  base_offline_folder + 'fNq' )
 

num_theta_functions = num_affine_components_A
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

fem_coordinates = gd.generate_fem_coordinates_from_list( number_of_fem_coordinates, "admissibleDofs_in.txt", sampling )
fem_coordinates = np.sort( fem_coordinates )

np.save("output_arrays/fem_coordinates", fem_coordinates)

number_of_output_coordinates = 60

fem_output_coordinates = gd.generate_fem_coordinates_from_list( number_of_output_coordinates, "admissibleDofs_out.txt", sampling )

fem_output_coordinates = np.sort( fem_output_coordinates )
np.save("output_arrays/fem_output_coordinates", fem_output_coordinates)


# #%%
# # number of selected parameters for the training
my_rb_manager.M_get_test = False
noise_magnitude = 0.0

X_train, y_train = gd.generate_fem_training_data( ns_train, fem_coordinates, fem_output_coordinates, 
                                                  my_rb_manager, 2, \
                                                  param_min, param_max, \
                                                  my_parameter_handler, noise_magnitude,
                                                  data_file = base_offline_folder + 'param.txt')

n_final_samples = ns_train

# if n_final_samples > ns_train:
#     X_train, y_train = gd.expand_train_with_rb( X_train, y_train, n_final_samples, \
#                                                 fem_coordinates,
#                                                 fem_output_coordinates, \
#                                                 my_rb_manager, 2, \
#                                                 param_min, param_max, \
#                                                 my_parameter_handler )


X_train = X_train[:n_final_samples,:]
y_train = y_train[:n_final_samples,:]
# #%%

# my_rb_manager.M_get_test = True
# noise_magnitude = 0.0

# X_test, y_test = gd.generate_fem_training_data( ns_test, fem_coordinates, fem_output_coordinates, 
#                                                   my_rb_manager, 2, \
#                                                   param_min, param_max, \
#                                                   my_parameter_handler, noise_magnitude,
#                                                   data_file = base_offline_folder + 'paramTest.txt' )

# #%%

######################################################################
################### TENSORFLOW mu_{in} -> mu_{pde} ###################
######################################################################

import elliptic_tensorflow_routines as etr

print( "######################################################################## ")
print( "################### TENSORFLOW mu_{in} -> mu_{pde} ################### ")
print( "######################################################################## ")

EPOCHS = 500
chosen_network_width = 'normal'

tf_model,history = etr.build_tensorflow_model( X_train, y_train, EPOCHS, network_width=chosen_network_width, lr = 0.001 )
etr.plot_history( history, 'exFeamat_ns500_Ninout60_lr0001')
# #%%

# ########################################################################
# ################# TENSORFLOW + PDEs mu_{in} -> mu_{pde} ################
# ########################################################################

# import elliptic_tensorflow_routines as etr
# # import pde_activation_mu_prediction_lifting as pa
# import pde_activation as pa


# print( "######################################################################## ")
# print( "############### TENSORFLOW + PDEs mu_{in} -> mu_{pde} ################## ")
# print( "######################################################################## ")

# EPOCHS = 50

# weight_mu = 0.0

# chosen_network_width = 'normal'

# my_pde_activation = pa.PDE_activation( my_rb_manager, number_of_output_coordinates + 2, fem_output_coordinates, 2, \
#                        param_min, param_max, [tf.identity, tf.sin, tf.cos], [0, 1, 1] )

# pde_model,history = etr.build_pde_tensorflow_model( X_train, y_train, EPOCHS, my_pde_activation, 2, weight_mu, \
#                                     network_width=chosen_network_width, optimizer='adam' )
# etr.plot_history( history )
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
