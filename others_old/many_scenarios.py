
from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import generate_data_elliptic as gd_elliptic
import elliptic_tensorflow_routines as etr
import numpy as np
from math import pi
import random

import tensorflow as tf
from tensorflow import keras
import numpy as np
print(tf.__version__)

tf.enable_eager_execution( )

#%%

# the domain Omega is (L0, L1)^2
L0 = 0; L1 = 1

#########################################################
#################### PROBLEM SETTING ####################
#########################################################

# Parameter ranges
alpha_min = 0.5; alpha_max = 1.0
phi_ext_min = 0.5; phi_ext_max = 1.0

myPFG = gd_elliptic.Parametrized_function_generator( )
myPFG.M_parameter_handler.assign_parameters_bounds( alpha_min, phi_ext_min, alpha_max, phi_ext_max )

print_locations = False
number_of_output_locations = 200
xy_output_locations = gd_elliptic.generate_locations( number_of_output_locations, L0, L1, print_locations )

ns_test = 1000
noise_magnitude = 0.0

number_of_locations_array = np.array( [100, 200, 400, 800, 1600] )
ns_array = np.array( [100, 200, 400, 800, 1600] )
num_parameters = myPFG.get_num_parameters( )

v_pde_alpha_errors = np.zeros( (len(number_of_locations_array), len(ns_array)) )
v_pde_phi_errors = np.zeros( (len(number_of_locations_array), len(ns_array)) )
v_pde_tot_errors = np.zeros( (len(number_of_locations_array), len(ns_array)) )

v_tf_alpha_errors = np.zeros( (len(number_of_locations_array), len(ns_array)) )
v_tf_phi_errors = np.zeros( (len(number_of_locations_array), len(ns_array)) )
v_tf_tot_errors = np.zeros( (len(number_of_locations_array), len(ns_array)) )

v_pde_loss = np.zeros( (len(number_of_locations_array), len(ns_array)) )
v_tf_loss = np.zeros( (len(number_of_locations_array), len(ns_array)) )
v_pde_mae = np.zeros( (len(number_of_locations_array), len(ns_array)) )
v_tf_mae = np.zeros( (len(number_of_locations_array), len(ns_array)) )


for iLoc in range( len( number_of_locations_array ) ):
    for iNs in range( len( ns_array ) ):
    
        print( "number of locations: %d --- number of snapshots: %d " % ( number_of_locations_array[iLoc], ns_array[iNs] ) )
        
        xy_locations = gd_elliptic.generate_locations( number_of_locations_array[iLoc], L0, L1, print_locations )
        
        X_train, y_train = gd_elliptic.generate_training_data( ns_array[iNs], xy_locations, xy_output_locations, myPFG, noise_magnitude )
        X_test, y_test = gd_elliptic.generate_training_data( ns_test, xy_locations, xy_output_locations, myPFG, noise_magnitude )

        print( "######################################################################## ")
        print( "############### TENSORFLOW + PDEs mu_{in} -> mu_{pde} ################## ")
        print( "######################################################################## ")
        
        EPOCHS = 2
        
        
        pde_model = etr.build_pde_tensorflow_model( X_train, y_train, EPOCHS, num_parameters, xy_output_locations )
        
        print( "######################################################################## ")
        print( "################### TENSORFLOW mu_{in} -> mu_{pde} ################### ")
        print( "######################################################################## ")
        
        tf_model = etr.build_tensorflow_model( X_train, y_train, EPOCHS )
        
        [loss, mae] = pde_model.evaluate(X_test, y_test, verbose=0)
        print("\n\nPDE Loss and mae are %f   %f" % (loss, mae))
        
        [tf_loss, tf_mae] = tf_model.evaluate(X_test, y_test, verbose=0)
        print("\n\nTF  Loss and mae for are %f   %f" % (tf_loss, tf_mae))
        
        pde_error_alpha, pde_error_phi, pde_error_test = etr.evaluate_model( pde_model, 'PDE', 
                                                                            X_train, y_train, X_test, y_test )
        
        tf_error_test_alpha, tf_error_test_phi, tf_error_test = etr.evaluate_model( tf_model, 'TF', 
                                                                                   X_train, y_train, X_test, y_test )
        
        etr.compare_two_model( pde_model, tf_model, 'PDE', 'TF', X_train, y_train, X_test, y_test )
        
        if tf_error_test_alpha > pde_error_alpha:
            print( "PDE wins for alpha    !     PDE: %f vs TF: %f " %( pde_error_alpha, tf_error_test_alpha ) )
        else:
            print( "TF  wins for alpha    !     PDE: %f vs TF: %f " %( pde_error_alpha, tf_error_test_alpha ) )
        
        if tf_error_test_phi > pde_error_phi:
            print( "PDE wins for phi      !     PDE: %f vs TF: %f " %( pde_error_phi, tf_error_test_phi ) )
        else:
            print( "TF  wins for phi      !     PDE: %f vs TF: %f " %( pde_error_phi, tf_error_test_phi ) )
        
        if tf_error_test > pde_error_test:
            print( "PDE wins in general   !     PDE: %f vs TF: %f " %( pde_error_test, tf_error_test ) )
        else:
            print( "TF  wins in general   !     PDE: %f vs TF: %f " %( pde_error_test, tf_error_test ) )

        v_pde_alpha_errors[iLoc, iNs] = pde_error_alpha
        v_pde_phi_errors[iLoc, iNs] = pde_error_phi
        v_pde_tot_errors[iLoc, iNs] = pde_error_test
        
        v_tf_alpha_errors[iLoc, iNs] = tf_error_test_alpha
        v_tf_phi_errors[iLoc, iNs] = tf_error_test_phi
        v_tf_tot_errors[iLoc, iNs] = tf_error_test

        v_pde_loss[iLoc, iNs] = loss
        v_tf_loss[ iLoc, iNs] = tf_loss
        v_pde_mae[ iLoc, iNs] = mae
        v_tf_mae[  iLoc, iNs] = tf_mae





