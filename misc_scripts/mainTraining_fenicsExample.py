from __future__ import absolute_import, division, print_function
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def youngPoisson2lame(param):
    
    n = len(param)
    paramNew = np.zeros((n,2))
    for i in range(n):
        nu = param[i,0]
        E = param[i,1]
        
        lamb = nu * E/((1. - 2.*nu)*(1.+nu))
        mu = E/(2.*(1. + nu))
        
        paramNew[i,0] = lamb
        paramNew[i,1] = mu

    
    return paramNew        
    

def youngPoisson2lame_minmax(nu_min,nu_max,E_min,E_max):
    
    lamb1 = nu_min * E_min/((1. - 2.*nu_min)*(1.+nu_min))
    lamb2 = nu_max * E_min/((1. - 2.*nu_max)*(1.+nu_max))
    lamb3 = nu_min * E_max/((1. - 2.*nu_min)*(1.+nu_min))
    lamb4 = nu_max * E_max/((1. - 2.*nu_max)*(1.+nu_max))
    
    mu1 = E_min/(2.*(1. + nu_min))
    mu2 = E_min/(2.*(1. + nu_max))
    mu3 = E_max/(2.*(1. + nu_min))
    mu4 = E_max/(2.*(1. + nu_max))
    
    lamb = np.array([lamb1,lamb2,lamb3,lamb4])
    mu = np.array([mu1,mu2,mu3,mu4])
    
    return lamb.min(),lamb.max(), mu.min(), mu.max()

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
import elliptic_tensorflow_routines as etr
import pickle

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

base_offline_folder = './rb_bar_lame/'

# Parameter ranges
nu_min = 0.25;  nu_max = 0.35
E_min = 40.0; E_max = 60.0

lamb_min, lamb_max, mu_min, mu_max = youngPoisson2lame_minmax(nu_min,nu_max,E_min,E_max)
# np.savetxt(base_offline_folder + 'paramLame.txt', youngPoisson2lame(np.loadtxt(base_offline_folder + 'param.txt')))
# np.savetxt(base_offline_folder + 'paramTestLame.txt', youngPoisson2lame(np.loadtxt(base_offline_folder + 'paramTest.txt')))

# in total ns = 500
ns_train = 200
ns_test  = 100

param_min = np.array([lamb_min, lamb_max])
param_max = np.array([mu_min, mu_max])
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

my_rb_manager.build_rb_approximation( ns_train, 1.e-10 ) # The same used in the Rb generation 

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
                                                  my_rb_manager, 2, \
                                                  param_min, param_max, \
                                                  my_parameter_handler, noise_magnitude,
                                                  data_file = base_offline_folder + 'param.txt')

n_final_samples = 10000

if n_final_samples > ns_train:
    X_train, y_train = gd.expand_train_with_rb( X_train, y_train, n_final_samples, \
                                                fem_coordinates,
                                                fem_output_coordinates, \
                                                my_rb_manager, 2, \
                                                param_min, param_max, \
                                                my_parameter_handler )


X_train = X_train[:n_final_samples,:]
y_train = y_train[:n_final_samples,:]

print(X_train.shape)
print(y_train.shape)

np.random.shuffle(X_train)
np.random.shuffle(y_train)


print(X_train.shape)
print(y_train.shape)
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



# runs = 10

# for i in range(10):
#     Run_id = str(1 + i)
#     print( "######################################################################## ")
#     print( "################### TENSORFLOW mu_{in} -> mu_{pde} ################### ")
#     print( "######################################################################## ")

#     EPOCHS = 500
#     chosen_network_width = 'constant'

#     tf_model,history = etr.build_tensorflow_model( X_train, y_train, EPOCHS, network_width=chosen_network_width, lr = 0.001 )
#     etr.plot_history(history)

#     with open('./saves_exFenics_ns2000_Ninout60_lr0001/historyModel_' + Run_id + '.dat', 'wb') as f:
#         pickle.dump(history.history, f)
        
    
#     tf_model.save('./saves_exFenics_ns2000_Ninout60_lr0001model_' + Run_id + '.hd5')
# #%%


# input()


########################################################################
################# TENSORFLOW + PDEs mu_{in} -> mu_{pde} ################
########################################################################

import pde_activation_elasticity as pa


# print( "######################################################################## ")
# print( "############### TENSORFLOW + PDEs mu_{in} -> mu_{pde} ################## ")
# print( "######################################################################## ")

Run_id = ''
EPOCHS = 100

weight_mu = 0.0

chosen_network_width = 'constant'

my_pde_activation = pa.PDE_activation( my_rb_manager, number_of_output_coordinates + 2, fem_output_coordinates, 2, \
                        param_min, param_max, [tf.identity, tf.identity], [0, 1] )

pde_model,history = etr.build_pde_tensorflow_model( X_train, y_train, EPOCHS, my_pde_activation, 2, weight_mu, \
                                    network_width=chosen_network_width, optimizer='adam' , lr = 0.1)
    
etr.plot_history( history, './saves_pde/loss_ns10000_8')

with open('./saves_pde/historyModel_' + Run_id + '.dat', 'wb') as f:
    pickle.dump(history.history, f)
    

pde_model.save('./saves_pde/' + Run_id + '.hd5')

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
