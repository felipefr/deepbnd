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

def create_rb_manager(exp_tol_rb):
    cpp_library_path = '/net/smana3/vol/vol1/cmcs/pegolott/deeplearning_pdes/lifev-pyorb/build/libpyorb-lifev-api.so'
    
    my_cpp_engine_manager = mee.external_engine_manager( 'cpp', cpp_library_path )
    my_cpp_engine_manager.start_engine( )
    my_cpp_external_engine = my_cpp_engine_manager.get_external_engine( )
    
    # Parameter ranges
    mu0_min = 0.5; mu0_max = 10.
    mu1_min = 0; mu1_max = 30.
    
    ns_train = 350
    
    param_min = np.array([mu0_min, mu1_min])
    param_max = np.array([mu0_max, mu1_max])
    
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
    
    name_str = "affine_advection"
    
    tol_rb = 10**(-exp_tol_rb)
    tol_rb_string = str(tol_rb) + 'tol'
    
    # building the RB manager
    my_rb_manager = rm.RbManager( my_affine_decomposition, my_aap )
    
    my_rb_manager.save_offline_structures( "rb_structures/snapshots_" + name_str + '.txt', \
                                           "rb_structures/basis_" + name_str + '_' + tol_rb_string + '.txt', \
                                           "rb_structures/rb_affine_components_" + name_str + '_' + tol_rb_string, \
                                           "rb_structures/offline_parameters.data" )
    
    snapshots_file = base_offline_folder + 'snapshots_affine_advection_lifting.txt'
    my_rb_manager.import_snapshots_matrix( snapshots_file )
    
    my_rb_manager.import_snapshots_parameters( base_offline_folder + 'offline_parameters.data' )
    
    my_rb_manager.build_rb_approximation( ns_train, tol_rb )
    
for i in range(4,5):
    create_rb_manager(i)