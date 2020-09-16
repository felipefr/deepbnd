from __future__ import absolute_import, division, print_function
import numpy as np
import pickle

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


import pyorb_core.tpl_managers.external_engine_manager as mee

cpp_library_path = '/net/smana3/vol/vol1/cmcs/pegolott/deeplearning_pdes/lifev-pyorb/build/libpyorb-lifev-api.so'

my_cpp_engine_manager = mee.external_engine_manager( 'cpp', cpp_library_path )
my_cpp_engine_manager.start_engine( )
my_cpp_external_engine = my_cpp_engine_manager.get_external_engine( )

# Parameter ranges
mu0_min = 0.5; mu0_max = 10.
mu1_min = 0; mu1_max = 30.

ns_train = 350
ns_test  = 600

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

my_aap.configure_fom( my_cpp_external_engine, fom_specifics )

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


number_of_output_coordinates = 1000

fem_output_coordinates = gd.generate_fem_coordinates_from_list( number_of_output_coordinates, "input_arrays/set1_yplane.txt", \
                                                      sampling )

fem_output_coordinates = np.sort( fem_output_coordinates )

my_rb_manager.M_get_test = False
noise_magnitude = 0.0

X_train, y_train = gd.generate_fem_training_data( ns_train, fem_coordinates, fem_output_coordinates, 
                                                  my_rb_manager, 2, \
                                                  param_min, param_max, \
                                                  my_parameter_handler, noise_magnitude,
                                                  data_file = base_offline_folder + 'offline_parameters.data' )

n_final_samples = 20000
n_final_samples = 350
if n_final_samples > ns_train:
    X_train, y_train = gd.expand_train_with_rb( X_train, y_train, n_final_samples, \
                                                fem_coordinates,
                                                fem_output_coordinates, \
                                                my_rb_manager, 2, \
                                                param_min, param_max, \
                                                my_aap )

X_train = X_train[:n_final_samples,:]
y_train = y_train[:n_final_samples,:]

my_rb_manager.M_get_test = True

noise_magnitude = 0.0

X_test, y_test = gd.generate_fem_training_data( ns_test, fem_coordinates, fem_output_coordinates, 
                                                  my_rb_manager, 2, \
                                                  param_min, param_max, \
                                                  my_parameter_handler, noise_magnitude,
                                                  data_file = base_offline_folder + 'test_parameters.data' )


# use this to generate y_complete
X_test, y_test = gd.generate_fem_training_data( ns_test, fem_coordinates, np.array(range(12414)), 
                                                  my_rb_manager, 2, \
                                                  param_min, param_max, \
                                                  my_parameter_handler, noise_magnitude,
                                                  data_file = base_offline_folder + 'test_parameters.data' )

np.save( "input_arrays/y_complete", y_test )

tol_rb = 1e-7

#np.save("output_arrays/fem_output_coordinates", fem_output_coordinates)
#np.save("output_arrays/fem_coordinates", fem_coordinates)
#np.save( "input_arrays/X_train_" + str(n_final_samples) + "samples_" + str(tol_rb) + "tol", X_train )
#np.save( "input_arrays/y_train_" + str(n_final_samples) + "samples_" + str(tol_rb) + "tol", y_train )
#np.save( "input_arrays/X_test_" + str(n_final_samples) + "samples_" + str(tol_rb) + "tol", X_test )
#np.save( "input_arrays/y_test_" + str(n_final_samples) + "samples_" + str(tol_rb) + "tol", y_test)