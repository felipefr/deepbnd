
import numpy as np
from math import pi
import random
import matplotlib.pyplot as plt

from pathlib import Path

class Parameter_handler:
    """a class for handling the parameters."""

    def __init__( self ):
        return

    def assign_parameters_bounds( self, _param_min, _param_max ):
        self.M_param_min = _param_min
        self.M_param_max = _param_max
        self.M_param     = np.zeros( _param_min.shape )
        self.M_num_parameters = _param_min.shape[0]

    def assign_parameters( self, _param ):
        
        assert self.M_num_parameters == _param.shape[0]
        self.M_param = _param

    def print_parameters( self ):
        print( "Numberof parameters : %d " % self.M_num_parameters )
        print( "The current parameter is: " )
        print( self.M_param )

    def generate_parameter( self ):
        # generate numbers between 0 and 1
        assert self.M_num_parameters > 0
        
        for iP in range( self.M_num_parameters ):
            pRandom = float( random.randint(0,10000) ) / 10000.0
            self.M_param[iP] = self.M_param_min[iP] + pRandom * ( self.M_param_max[iP] - self.M_param_min[iP] )
        
    def get_parameter( self ):
        return self.M_param

    def get_parameter_vector( self ):
        return self.M_param

    def get_num_parameters( self ):
        return self.M_num_parameters

    def get_min_parameters( self ):
        return self.M_param_min

    def get_max_parameters( self ):
        return self.M_param_max

    def get_range_parameters( self ):
        return self.M_param_max - self.M_param_min


    M_param_min = np.zeros( 0 )
    M_param_max = np.zeros( 0 )
    M_param     = np.zeros( 0 )
    M_num_parameters = 0
    

def generate_locations( number_of_locations_per_direction, L0, L1, print_locations ):
    x_locations = np.zeros( number_of_locations_per_direction )
    y_locations = np.zeros( number_of_locations_per_direction )

    for iLoc in range( number_of_locations_per_direction ):
        x_locations[ iLoc ] = float( random.randint(0,10000) ) / 10000.0
        y_locations[ iLoc ] = float( random.randint(0,10000) ) / 10000.0

#    xy_locations = list( zip( x_locations, y_locations ) ) # construct iterable with tuples of locations

    xy_locations = np.zeros( ( number_of_locations_per_direction, 2 ) )
    xy_locations[:, 0] = x_locations
    xy_locations[:, 1] = y_locations

    if print_locations == True:
        plt.figure();
        plt.plot( x_locations, y_locations, '.' );
        plt.grid( False );
        plt.show();

    # print( "Locations: in %d points \n" % len(xy_locations) )
    # print( xy_locations )

    return xy_locations

def generate_training_parameters( ns, parameter_handler, output_file_name="" ):
    
    num_parameters = parameter_handler.get_num_parameters( )
    
    chosen_parameters = np.zeros( ( ns, num_parameters ) )
    my_output_file = Path( output_file_name )
    
    read_parameters_size = 0
    
    if my_output_file.is_file():
        print( "The file name '%s' provided exists, therefore we recover from here the parameters " % output_file_name )
        read_parameters = np.loadtxt( output_file_name )

        read_parameters_size = read_parameters.shape[0]
    
        if read_parameters_size >= ns:
            print( "There are too many parameters here! I cut the matrix ..." )
            chosen_parameters = read_parameters[0:ns, :]
        else:
            chosen_parameters[0:read_parameters_size, :] = read_parameters
    else:
        print( "We need to build the paramters from scratch since the file '%s' does not exist! " % output_file_name )
    
    if read_parameters_size < ns:
        print( "We need more parameters here, and specifically %d! I build some new of them ..." % (ns - read_parameters_size ) )

        for iNs in range( read_parameters_size, ns ):
            parameter_handler.generate_parameter( )
#            parameter_handler.print_parameters( )
            chosen_parameters[iNs, :] = parameter_handler.get_parameter_vector( )
            
        if output_file_name != "":
            output_file = open( output_file_name, 'a+' )
            
            for iNs in range( read_parameters_size, ns ):
                for iP in range( num_parameters ):
                    output_file.write( "%.6g" % chosen_parameters[iNs, iP] )
    
                    if iP < num_parameters - 1:
                        output_file.write( " " % chosen_parameters[iNs, iP] )
                    else:
                        output_file.write( "\n" % chosen_parameters[iNs, iP] )
        
            output_file.close( )
        
    return chosen_parameters




def generate_training_data( ns, xy_locations, xy_output_locations, parametrized_function_generator, noise_magnitude, 
                            data_file, printParam ):

    num_parameters = parametrized_function_generator.M_parameter_handler.get_num_parameters( )

    num_locations = xy_locations.shape[0]
    num_output_locations = xy_output_locations.shape[0]

    y_output = np.zeros( ( ns, num_parameters + num_output_locations ) )

    # measurements of the solution, should they be noised?
    u_ex_locations = np.zeros( (ns, num_locations) )

    y_output[:, -num_parameters:] = generate_training_parameters( ns, parametrized_function_generator.M_parameter_handler, data_file )
    
    for iNs in range( ns ):
#        parametrized_function_generator.M_parameter_handler.generate_parameter( )
#        y_output[iNs, -2:] = parametrized_function_generator.M_parameter_handler.get_parameter_vector( )

        parametrized_function_generator.assign_parameters( y_output[iNs, -num_parameters:] )
        
        # collecting the data which will be the input of the model
        for iLoc in range( num_locations ):
            u_ex_locations[iNs, iLoc] = parametrized_function_generator.parametrized_function_list( xy_locations[iLoc, 0], xy_locations[iLoc, 1] )

        for iLoc in range( num_output_locations ):
            y_output[iNs, iLoc ] = parametrized_function_generator.parametrized_function_list( xy_output_locations[iLoc, 0], xy_output_locations[iLoc, 1] )

    min_param = parametrized_function_generator.get_min_parameters( )
    max_param = parametrized_function_generator.get_max_parameters( )

    y_output[:, -num_parameters:] = ( y_output[:, -num_parameters:] - min_param ) / (max_param - min_param)

    u_ex_noised = u_ex_locations + noise_magnitude * np.random.normal( 0, 1, np.shape( u_ex_locations ) ) * u_ex_locations
    y_output_noised = ( 1 + noise_magnitude * np.random.normal( 0, 1, np.shape( y_output ) ) ) * y_output

    # printing the first two parameters when needed
    if printParam:
        plt.figure( )
        plt.scatter( y_output[:, -2:-1], y_output[:, -1:] )
        
    return u_ex_noised, y_output_noised



# this functions aims at selecting the elements of the fem arrays used for the network, 
# the coordinates are in the interval [min_coord, max_coord)
def generate_fem_coordinates( number_of_fem_coordinates, min_coord, max_coord, sampling='random', dof_per_direction=0 ):
    
    if sampling == 'random':

        fem_locations = np.zeros( number_of_fem_coordinates )
    
        for iCoord in range( number_of_fem_coordinates ):
            fem_locations[ iCoord ] = random.randint(min_coord,max_coord)

    elif sampling == 'tensorial':
        
        fem_locations = np.zeros( number_of_fem_coordinates[0] * number_of_fem_coordinates[1] )

        jump_x = np.ceil( float(dof_per_direction) / float( number_of_fem_coordinates[0] + 1 ) )
        jump_from_border_x = np.floor( ( float(dof_per_direction) - jump_x * float( number_of_fem_coordinates[0] + 1 ) ) / 2. )

        print( jump_from_border_x )

        jump_y = np.ceil( float(dof_per_direction) / float( number_of_fem_coordinates[1] + 1 ) )
        jump_from_border_y = np.floor( ( float(dof_per_direction) - jump_y * float( number_of_fem_coordinates[1] + 1 ) ) / 2. )
        
        print('Choosing tensorial grid selection, with jumps %f, %f and jumps from border %f, %f' \
            % (jump_x, jump_y, jump_from_border_x, jump_from_border_y) )
        
        fem_location_counter = 0;
        
        for iX in range( number_of_fem_coordinates[0] ):
            for iY in range( number_of_fem_coordinates[1] ):
                
                fem_locations[fem_location_counter] = jump_from_border_x  + dof_per_direction * jump_from_border_y \
                                                    + (iY+1) * jump_y * dof_per_direction \
                                                    + (iX+1) * jump_x
                
                fem_location_counter = fem_location_counter + 1
        
    return fem_locations.astype( int )

# this functions aims at selecting the elements of the fem arrays used for the network, given
# a datfile containing ammisible components
def generate_fem_coordinates_from_list( number_of_fem_coordinates, filename, sampling='random' ):
    
    coords = np.loadtxt ( filename );
    ncoords = coords.shape[0]
    
    if sampling == 'random':

        fem_locations = np.zeros( number_of_fem_coordinates )
    
        for iCoord in range( number_of_fem_coordinates ):
            rind = random.randint(0,ncoords)
            fem_locations[ iCoord ] = coords[ rind ]

        
    return fem_locations.astype( int )

def generate_fem_training_data( ns, fem_coordinates, fem_output_coordinates, snapshot_collector, _num_parameters, \
                                min_parameter, max_parameter, \
                                parameter_handler, noise_magnitude, data_file=None, printParam=False ):

    num_parameters = _num_parameters

    print( 'Num of paramters is %d' % num_parameters )

    num_locations = fem_coordinates.shape[0]
    num_output_locations = fem_output_coordinates.shape[0]

    y_output = np.zeros( ( ns, num_parameters + num_output_locations ) )

    # measurements of the solution, should they be noised?
    u_ex_locations = np.zeros( (ns, num_locations) )

    if data_file != None:
        y_output[:, -num_parameters:] = generate_training_parameters( ns, parameter_handler, data_file )
    
    for iNs in range( ns ):
        
        u_ex_locations[iNs, :] = snapshot_collector.get_snapshot_function( iNs, fem_coordinates )

        y_output[iNs, 0:num_output_locations] = snapshot_collector.get_snapshot_function( iNs, fem_output_coordinates )

    min_param = min_parameter[0:num_parameters]
    max_param = max_parameter[0:num_parameters]

    y_output[:, -num_parameters:] = ( y_output[:, -num_parameters:] - min_param ) / (max_param - min_param)
    # y_output[:, -num_parameters:] = 2. / (max_param - min_param) * y_output[:, -num_parameters:]  - (min_param + max_param) / (max_param - min_param)

    u_ex_noised = u_ex_locations + noise_magnitude * np.random.normal( 0, 1, np.shape( u_ex_locations ) ) * u_ex_locations
    y_output_noised = ( 1. + noise_magnitude * np.random.normal( 0, 1, np.shape( y_output ) ) ) * y_output

    # printing the first two parameters when needed
    if printParam:
        plt.figure( )
        plt.scatter( y_output[:, -2:-1], y_output[:, -1:] )
        
    return u_ex_noised, y_output_noised

def expand_train_with_rb( X_train, y_train, n_final_samples, \
                          fem_coordinates,
                          fem_output_coordinates, \
                          rb_manager, num_parameters, \
                          param_min, param_max, \
                          fom_problem ):
     
    n_cur_samples = X_train.shape[0]
    samples_to_take = n_final_samples - n_cur_samples
    
    X_out = np.zeros( (n_final_samples, X_train.shape[1]) )
    y_out = np.zeros( (n_final_samples, y_train.shape[1]) )
    
    X_out[:n_cur_samples,:] = X_train
    y_out[:n_cur_samples,:] = y_train
    
    
    for iis in range( samples_to_take ) :
        fom_problem.generate_parameter()
        new_param = fom_problem.get_parameter()


        param_string = 'Param number ' + str(iis) + ' = [ '
        for jp in range( param_min.shape[0] ):
            param_string = param_string + str( new_param[jp] ) + ' , '
            
        param_string = param_string[:-2] + ']'
            
        print( param_string )

        rb_manager.solve_reduced_problem( np.array(new_param) )
        rb_manager.reconstruct_fem_solution( rb_manager.M_un )
        new_snap = rb_manager.M_utildeh
        new_snap = np.reshape( new_snap, (new_snap.shape[0], ))
        X_out[iis + n_cur_samples,:] = new_snap[fem_coordinates]
        y_out[iis + n_cur_samples,:-num_parameters] = new_snap[fem_output_coordinates]
        y_out[iis + n_cur_samples,-num_parameters:] = (new_param - param_min) / (param_max - param_min)

        
    return X_out, y_out









#
