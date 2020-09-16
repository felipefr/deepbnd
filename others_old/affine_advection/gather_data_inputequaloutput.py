from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import tensorflow as tf
import os, sys
import time 
sys.path.insert(0, '..')
sys.path.insert(0, '../../../pyorb/')
sys.path.insert(0, '../../../pyorb/examples/affine_advection_cpp')

import numpy as np
import affine_advection as aap
import pyorb_core.rb_library.rb_manager as rm
import pyorb_core.rb_library.affine_decomposition as ad
import pyorb_core.pde_problem.parameter_handler as ph
import random
import elliptic_tensorflow_routines as etr
import pde_activation as pa

results_folder = 'results_inputequaloutput'


def create_folder( name_folder ):
    if not os.path.exists( name_folder ):
        os.mkdir( name_folder )
        return True
    else:
        return False

        
def save_history( name_folder, history ):
    np.save( name_folder + 'loss', history.history['loss'] )
    np.save( name_folder + 'mean_absolute_error', history.history['mean_absolute_error'] )
    np.save( name_folder + 'val_loss', history.history['val_loss'] )
    np.save( name_folder + 'val_mean_absolute_error', history.history['val_mean_absolute_error'] )

def write_value_to_file( name_file, value ):
    with open( name_file, "w+" ) as f:
        f.write( "%f" % value )

def save_test_result( name_folder, mat ):
    np.save( name_folder + 'test', mat )
    
def compute_error_on_u ( y_test, res, out_coordinates ):
    model_errors = y_test - res
    ns_test = y_test.shape[0]
    error_test_local = np.zeros( ns_test )
    for ii in range( ns_test ):
        error_test_local[ii] = np.sqrt( np.sum( model_errors[ii, :-2] * 
                                                model_errors[ii, :-2] ) )
    return np.mean( error_test_local )
    
def compute_errors_on_parameters( y_test, res ):
    model_errors = y_test - res
    error_test_parameters = np.zeros( (2,1) )
    for iP in range( 2 ):
        error_test_parameters[iP] = np.mean( np.sqrt( model_errors[:, -2+iP] * model_errors[:, -2+iP] ) )
        
    return error_test_parameters

def save_size( name_file, model ):
    size = 0
    for i in range( len( model.get_weights() ) ):
        size = size + model.get_weights()[i].size
    write_value_to_file( name_file, size)

def run_pipeline( num_inputs, num_outputs, rb_tol, n_samples, network_type, epochs ):
    name_folder = '/ni' + str(num_inputs) + '_no' + str(num_outputs) + '_rbtol' + str(rb_tol) + '_ns' + str(n_samples) + '_nt' + network_type + '_ep' + str(epochs) + '/'
    name_folder = results_folder + name_folder
    status = create_folder( name_folder )
    if not status:
        return
    create_folder( name_folder + 'pde' )

    X_train = np.load('input_arrays/X_train_20000samples_1e-07tol.npy')
    y_train = np.load('input_arrays/y_train_20000samples_1e-07tol.npy')
    X_test = np.load('input_arrays/X_test_20000samples_1e-07tol.npy')
    y_test = np.load('input_arrays/y_test_20000samples_1e-07tol.npy')
        
    # Parameter ranges
    mu0_min = 0.5; mu0_max = 10.
    mu1_min = 0; mu1_max = 30.
    
    param_min = np.array([mu0_min, mu1_min])
    param_max = np.array([mu0_max, mu1_max])
    
    my_parameter_handler = ph.Parameter_handler( )
    my_parameter_handler.assign_parameters_bounds( param_min, param_max )
    
    my_aap = aap.affine_advection_problem( my_parameter_handler )
    
    num_affine_components_A = 3
    num_affine_components_f = 3
    
    # defining the affine decomposition structure
    my_affine_decomposition = ad.AffineDecompositionHandler( )
    my_affine_decomposition.set_Q( num_affine_components_A, num_affine_components_f )    # number of affine terms
    
    # building the RB manager
    my_rb_manager = rm.RbManager( my_affine_decomposition, my_aap )
        
    my_rb_manager.import_basis_matrix( 'rb_structures/basis_affine_advection_' + str(rb_tol) + 'tol.txt' )
    my_affine_decomposition.import_rb_affine_matrices( 'rb_structures/rb_affine_components_affine_advection_' + str(rb_tol) + 'tol_A' )
    my_affine_decomposition.import_rb_affine_vectors(  'rb_structures/rb_affine_components_affine_advection_' + str(rb_tol) + 'tol_f' )
    
    new_locations_input = np.load( "input_arrays/shuffled_components.npy" )
    new_locations_input = new_locations_input[:num_inputs]
    new_locations_output = np.load( "input_arrays/shuffled_components.npy" )
    new_locations_output = new_locations_output[:num_outputs]

    fem_coordinates = np.load('output_arrays/fem_coordinates.npy')    
    fem_coordinates = fem_coordinates[ new_locations_input ]
    np.save( name_folder + 'cut_fem_coordinates', fem_coordinates )

    fem_output_coordinates = np.load('output_arrays/fem_output_coordinates.npy')  
    fem_output_coordinates = fem_output_coordinates[ new_locations_output ]
    np.save( name_folder + 'cut_fem_output_coordinates', fem_output_coordinates )
    
    actual_nlo = np.zeros( (num_outputs + 2,) )
    actual_nlo[0:num_outputs] = new_locations_output
    actual_nlo[-2] = 1000
    actual_nlo[-1] = 1001
    X_train = X_train[0:n_samples,new_locations_input]
    X_test = X_test[:,new_locations_input]
    y_train = y_train[0:n_samples,actual_nlo.astype(int)]
    y_test = y_test[:,actual_nlo.astype(int)]
    
    y_train[:,:-2] = X_train
    y_test[:,:-2] = X_test
    fem_output_coordinates = fem_coordinates
     
    n_runs = 10
    for j in range(n_runs):
        pde_folder = name_folder + 'pde/' + str(j) + '/'
        create_folder( pde_folder )
    
        my_pde_activation = pa.PDE_activation( my_rb_manager, num_outputs + 2, fem_output_coordinates, 2, \
                               param_min, param_max, [tf.identity, tf.sin, tf.cos], [0, 1, 1] )
        start = time.time()
        pde_model,history = etr.build_pde_tensorflow_model( X_train, y_train, epochs, my_pde_activation, 2, 0.0, \
                                            network_width=network_type, optimizer='adam' ) 
        save_size( pde_folder + "size.txt", pde_model )
        end = time.time()
        write_value_to_file( pde_folder + "rt.txt", end - start )
        save_history( pde_folder, history )

        res = pde_model.predict( X_test )
        save_test_result( pde_folder, res)
                
create_folder( results_folder )
inout_range = [ [20,20], [40,40], [100,100] ]
tols = [ 1e-5 ]
nsamples = [20000 ]
epochs = 500
for i in inout_range:
    for insamples in nsamples:
        for itols in tols:
            run_pipeline( i[0], i[1], itols, insamples, 'constant', epochs )
