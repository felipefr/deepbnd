from __future__ import absolute_import, division, print_function
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
import random

import pyorb_core.tpl_managers.external_engine_manager as mee

tols = [1e-4, 1e-5, 1e-6, 1e-7]
tols = [1e-7]
def generate(  Y, fem_output_coordinates, rb_manager, parameters ):
     
    n_cur_samples = X_train.shape[0]
    samples_to_take = n_final_samples - n_cur_samples
    
    X_out = np.zeros( (n_final_samples, X_train.shape[1]) )
    y_out = np.zeros( (n_final_samples, y_train.shape[1]) )
    
    X_out[:n_cur_samples,:] = X_train
    y_out[:n_cur_samples,:] = y_train
    
    
    for iis in range( samples_to_take ) :
            
        param_string = param_string[:-2] + ']'
            
        print( param_string )

        rb_manager.solve_reduced_problem( np.array(new_param) )
        rb_manager.reconstruct_fem_solution( rb_manager.M_un )
        new_snap = rb_manager.M_utildeh
        new_snap = np.reshape( new_snap, (new_snap.shape[0], ))
        X_out[iis + n_cur_samples,:] = new_snap[fem_coordinates]

        
    return X_out, y_out

for t in tols:

    # Parameter ranges
    mu0_min = 0.5; mu0_max = 10.
    mu1_min = 0; mu1_max = 30.
    
    ns_test  = 150
    
    param_min = np.array([mu0_min, mu1_min])
    param_max = np.array([mu0_max, mu1_max])
    num_parameters = param_min.shape[0]
    
    my_parameter_handler = ph.Parameter_handler( )
    my_parameter_handler.assign_parameters_bounds( param_min, param_max )
    
    # define the fem problem
    import affine_advection as aap
    
    my_aap = aap.affine_advection_problem( my_parameter_handler )
    
    fom_specifics = {
            'model'             : 'affine_advection', \
            'datafile_path'     : './simulation_data/data'}
    
    base_offline_folder = '/net/smana3/vol/vol1/cmcs/pegolott/deeplearning_pdes/pyorb/examples/affine_advection_cpp/offline_affine_advection_lifting/'
    
    print( 'The base offline folder is %s ' % base_offline_folder )
    
    num_affine_components_A = 3
    num_affine_components_f = 3
    
    # defining the affine decomposition structure
    my_affine_decomposition = ad.AffineDecompositionHandler( )
    my_affine_decomposition.set_Q( num_affine_components_A, num_affine_components_f )    # number of affine terms
    
    # building the RB manager
    my_rb_manager = rm.RbManager( my_affine_decomposition, my_aap )
    
    affine_components_folder = 'rb_structures'
    
    my_rb_manager.import_basis_matrix( 'rb_structures/basis_affine_advection_1e-07tol.txt' )
    my_affine_decomposition.import_rb_affine_matrices( 'rb_structures/rb_affine_components_affine_advection_1e-07tol_A' )
    my_affine_decomposition.import_rb_affine_vectors(  'rb_structures/rb_affine_components_affine_advection_1e-07tol_f' )
    snapshots_file = base_offline_folder + 'snapshots_affine_advection_lifting.txt'
    my_rb_manager.import_snapshots_matrix( snapshots_file )
    
    my_rb_manager.import_snapshots_parameters( base_offline_folder + 'offline_parameters.data' )
    
    num_theta_functions = num_affine_components_A
    train_parameters = my_rb_manager.get_offline_parameters( )
    
    my_rb_manager.import_test_parameters( base_offline_folder + 'test_parameters.data' )
    my_rb_manager.import_test_snapshots_matrix( base_offline_folder + 'snapshots_test.txt' )
    my_rb_manager.M_N = 38
    
    test_parameters = my_rb_manager.get_test_parameter_matrix( )
    
    import generate_data as gd
    
    n_dofs = my_rb_manager.get_snapshots_matrix().shape[0]
    
    number_of_fem_coordinates = 1000
    
    sampling = 'random'
    
    fem_coordinates = gd.generate_fem_coordinates_from_list( number_of_fem_coordinates, "input_arrays/set0_yplane.txt", \
                                                  sampling )
    fem_coordinates = np.sort( fem_coordinates )
    
    # np.save("output_arrays/fem_coordinates", fem_coordinates)
    
    number_of_output_coordinates = 1000
    
    fem_output_coordinates = gd.generate_fem_coordinates_from_list( number_of_output_coordinates, "input_arrays/set1_yplane.txt", \
                                                          sampling )
    
    fem_output_coordinates = np.sort( fem_output_coordinates )
    # np.save("output_arrays/fem_output_coordinates", fem_output_coordinates)
    
    my_rb_manager.M_get_test = True
    noise_magnitude = 0.0
    
    X, Y = gd.generate_fem_training_data( ns_test, fem_coordinates, fem_output_coordinates, 
                                          my_rb_manager, 2, \
                                          param_min, param_max, \
                                          my_parameter_handler, noise_magnitude,
                                          data_file = base_offline_folder + 'offline_parameters.data' )
        
    noise_magnitude = 0.0
    Yrb = generate( Y, fem_output_coordinates, rb_manager, parameters )