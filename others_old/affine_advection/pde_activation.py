import numpy as np
import tensorflow as tf
import sys
sys.path.insert(0, '..')

# class aimed at hosting the PDE activation in the network and anything needed to use it 

class PDE_activation:

    def __init__( self, _rb_manager, _number_of_output, _output_locations, _num_pde_activation_input, \
                  params_min, params_max, theta_fcts, mu_map ):
        
        self.M_params_min = params_min
        self.M_params_max = params_max       
        self.M_num_pde_activation_input = _num_pde_activation_input
        self.M_theta_fcts = theta_fcts
        self.M_mu_map = mu_map
        self.prepare_selectors( _rb_manager, _number_of_output, _output_locations )       
        self.import_affine_arrays( _rb_manager )
        
        return

    def prepare_selectors( self, _rb_manager, _number_of_output, _output_locations ):
        
        print( "Preparing selectors for handling the RB problem" )

        self.M_A_mu_selectors = []

        for iSelector in range( self.M_num_pde_activation_input ):
            A_np_selector = np.zeros( ( self.M_num_pde_activation_input, 1 ) )
            A_np_selector[iSelector, 0] = 1
            self.M_A_mu_selectors.append( tf.add( tf.zeros( A_np_selector.shape ), A_np_selector ) )
                        
        VT_np_output = np.transpose( _rb_manager.get_basis( _output_locations ) )
        self.M_N = VT_np_output.shape[0]
        self.M_VT_output = tf.convert_to_tensor( VT_np_output, dtype=tf.float32 )

        A_enlarge_mu_np = np.zeros( ( self.M_num_pde_activation_input, _number_of_output ) )

        for iSelector in range( self.M_num_pde_activation_input ):
            starting_point = -self.M_num_pde_activation_input+iSelector
            A_enlarge_mu_np[iSelector, starting_point] = 1
        
        self.M_A_enlarge_mu = tf.add( tf.zeros( ( self.M_num_pde_activation_input, _number_of_output ) ), A_enlarge_mu_np )

        number_of_output_locations = _output_locations.shape[0]
        print( "number_of_output_locations is %d and total number of output is %d " % ( number_of_output_locations, _number_of_output ) )

        A_enlarge_f_np = np.zeros( ( number_of_output_locations, _number_of_output ) )
        A_enlarge_f_np[:, np.arange(number_of_output_locations)] = np.identity( number_of_output_locations )
        self.M_A_enlarge_f = tf.convert_to_tensor( A_enlarge_f_np, dtype=tf.float32 )

        return
        
    def  import_affine_arrays( self, _rb_manager ):
        
        self.M_rb_affine_matrices = []
        self.M_rb_affine_vectors = []
        
        # importing rb arrays and expanding tensors to handle the computations of RB linear systems at once
        for iQa in range( _rb_manager.get_Qa( ) ):
            self.M_rb_affine_matrices.append( tf.convert_to_tensor( _rb_manager.get_rb_affine_matrix( iQa ), dtype=tf.float32 ) )
            self.M_rb_affine_matrices[iQa] = tf.expand_dims(self.M_rb_affine_matrices[iQa], 0 )

        for iQf in range( _rb_manager.get_Qf( ) ):
            self.M_rb_affine_vectors.append( tf.convert_to_tensor( _rb_manager.get_rb_affine_vector( iQf ), dtype=tf.float32 ) )
            self.M_rb_affine_vectors[iQf] = tf.expand_dims(self.M_rb_affine_vectors[iQf], 0 )
            self.M_rb_affine_vectors[iQf] = tf.expand_dims(self.M_rb_affine_vectors[iQf], 2 )

        return
    
    M_num_pde_activation_input = 0
    
    M_N = 0
    M_min_param   = np.zeros( 0 )
    M_range_param = np.zeros( 0 )

    M_VT_output = tf.zeros( (0,0) )
    
    M_A_mu_selectors = []
    
    M_A_enlarge_mu = tf.zeros( (0,0) )
    M_A_enlarge_f = tf.zeros( (0,0) )

    M_rb_affine_matrices = []
    M_rb_affine_vectors = []

    M_params_min = 0
    M_params_max = 0
        
    def pde_mu_solver( self, _computed_mu ):
       
        mu_length = tf.shape(_computed_mu)[0]
        
        ns = tf.ones( mu_length ) 
        ns = tf.reshape( ns, ( mu_length, 1, 1) )


        # store parameters in a bigger matrix, ready to be summed 
        pde_mu_solver_tf_output_param = tf.matmul( _computed_mu, self.M_A_enlarge_mu )
        
        rb_matrix_online = tf.zeros( self.M_rb_affine_matrices[0].shape )
        rb_rhs_online = tf.zeros( self.M_rb_affine_vectors[0].shape )
        
        n_affine_components = len( self.M_theta_fcts )
        
        for iQa in range( n_affine_components ):
            imu = self.M_mu_map[iQa]
            # rescaled_mu = tf.matmul(_computed_mu * ( self.M_params_max[imu] - self.M_params_min[imu])/2.0 + ( self.M_params_max[imu] + self.M_params_min[imu])/2.0,  self.M_A_mu_selectors[imu] )
            rescaled_mu = tf.matmul( self.M_params_min[imu] + ( self.M_params_max[imu] - self.M_params_min[imu] ) * _computed_mu, self.M_A_mu_selectors[imu] )
            if (iQa == 0):
                theta_mu = self.M_theta_fcts[iQa]( rescaled_mu )
            else:
                theta_mu = self.M_theta_fcts[iQa]( rescaled_mu / 180.0 * np.pi )
            rb_matrix_online = tf.add( rb_matrix_online,    \
                                       ( tf.expand_dims( theta_mu , 1 ) ) \
                                       * ( ns * self.M_rb_affine_matrices[iQa] ) )
            rb_rhs_online = tf.add( rb_rhs_online,    \
                           ( tf.expand_dims( theta_mu, 1 ) ) \
                           * ( ns * self.M_rb_affine_vectors[iQa] ) )
    
        rb_sol_online = tf.matrix_solve( rb_matrix_online, rb_rhs_online )

        # this provides the transpose of a matrix containing all the RB solutions, i.e. [u_n(mu_0), ..., u_n(mu_ns)]^T
        rb_sol_online_2 = tf.reshape( rb_sol_online, (mu_length, self.M_N ) )

        pde_mu_solver_tf_output_0 = tf.matmul( rb_sol_online_2, self.M_VT_output )

        pde_mu_solver_tf_output = tf.matmul( pde_mu_solver_tf_output_0, self.M_A_enlarge_f )

        pde_mu_solver_tf_output = tf.add( pde_mu_solver_tf_output, pde_mu_solver_tf_output_param )

        return pde_mu_solver_tf_output
     









#
