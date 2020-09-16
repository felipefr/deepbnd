
import tensorflow as tf
from tensorflow import keras
import numpy as np
import generate_data as gd
from math import pi


def construct_tf_std_model( number_of_inputs, number_of_output, network_width='normal'):
    
#    if not (weight_mu == 1 and weight_loc == 1):
#        if weight_mu == 1:
#            number_of_output = num_parameters
#        elif weight_loc == 1:
#            number_of_output = number_of_output - num_parameters
#        else:
#            raise ValueError('weight_mu and weight_loc cannot be both 0!')

    print( "TF network with network %s " % network_width )
    
    if network_width == 'normal':
        
        print( "TF network with network normal " )

        model = keras.Sequential([
        keras.layers.Dense( 1024, activation=tf.nn.relu,
                           input_shape=(number_of_inputs,)),
            keras.layers.Dense( 512, activation=tf.nn.relu),
            keras.layers.Dense(256, activation=tf.nn.relu),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense( number_of_output, activation=tf.nn.sigmoid )
      ])
    
    elif network_width == 'small':
        
        print( "TF network with network small " )

        model = keras.Sequential([
        keras.layers.Dense( 64, activation=tf.nn.relu,
                            input_shape=(number_of_inputs,)),
        keras.layers.Dense( 64, activation=tf.nn.relu),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense( number_of_output, activation=tf.nn.sigmoid )
      ])
        
    elif network_width == 'constant':
        print( "TF network with network constant " )

        model = keras.Sequential([
        keras.layers.Dense( 256, activation=tf.nn.relu,
                            input_shape=(number_of_inputs,)),
        keras.layers.Dense( 256, activation=tf.nn.relu),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense( number_of_output, activation=tf.nn.sigmoid )
      ])
    
    elif network_width == 'large':
        
        print( "TF network with network large " )

        model = keras.Sequential([
                keras.layers.Dense( 2048, activation=tf.nn.relu,
                           input_shape=(number_of_inputs,)),
            keras.layers.Dense( 2048, activation=tf.nn.relu),
            keras.layers.Dense(2048, activation=tf.nn.relu),
            keras.layers.Dense(2048, activation=tf.nn.relu),
            keras.layers.Dense( number_of_output, activation=tf.nn.sigmoid )
      ])
    elif network_width == 'modified_felipe':   
        Nneurons = 64
        print( "TF network with modified " )
        model = keras.Sequential([
            keras.layers.Dense( Nneurons, activation=tf.keras.activations.linear,
                                input_shape=(number_of_inputs,)),
            keras.layers.Dense( Nneurons, activation=tf.nn.relu),
            keras.layers.Dense(Nneurons, activation=tf.nn.leaky_relu),
            keras.layers.Dense(Nneurons, activation=tf.nn.relu),
            keras.layers.Dense(Nneurons, activation=tf.nn.relu),
            keras.layers.Dense( number_of_output, activation=tf.keras.activations.linear)])
    
#  optimizer = tf.train.RMSPropOptimizer(0.00001)
#  optimizer= tf.train.GradientDescentOptimizer( 0.01 )


    return model

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

    
import matplotlib.pyplot as plt


def plot_history(history,savefile=None):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(history.epoch, np.array(history.history['loss']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_loss']),
           label = 'Val loss')
  plt.yscale('log')
  plt.legend()
#  plt.ylim([0, ])
  
  if savefile != None:
      plt.savefig(savefile + '.png')
      np.savetxt(savefile + '_history_.txt', np.array(history.history['loss']) )
      
  plt.show( )



def build_tensorflow_model( X_train, y_train, EPOCHS, network_width='normal', lr = 0.0001 ):
    
    model = construct_tf_std_model( X_train.shape[1], y_train.shape[1], network_width)
    model.summary()
    
    optimizer= tf.train.AdamOptimizer(lr)
    # optimizer= tf.train.AdamOptimizer(learning_rate = lr, beta1=0.8, beta2=0.7, epsilon=0.01, use_locking=False)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
    
    batchsize = int((np.min([X_train.shape[0]/20,100])))
    # Store training stats
    history = model.fit(X_train, y_train, epochs=EPOCHS,
                        validation_split=0.2, verbose=1,
                        callbacks=[PrintDot()], batch_size = 32)
    
#    plot_history(history)

    return model, history

def build_tensorflow_model_weights( X_train, y_train, num_parameters, EPOCHS, network_width='normal', 
                                   lr = 1.e-4, decay = 1.e-2, w_l = 1.0, w_mu = 1.0):
    
    model = construct_tf_std_model( X_train.shape[1], y_train.shape[1], network_width)
    model.summary()
    
    # optimizer= tf.train.AdamOptimizer(lr)
    optimizer= keras.optimizers.Adam(learning_rate = lr)
    
    
    def scheduler(epoch):
        omega = np.sqrt(float(epoch/EPOCHS))
        rate = lr*(1.0 - omega) + omega*decay*lr
        print('learning_rate = ', rate)
        return rate
        
    decay_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
       
            
    npar = num_parameters
    def custom_loss(y_p, y_d):
        return tf.reduce_mean( w_l * tf.square(y_p[:, :-npar] - y_d[:, :-npar] )) + tf.reduce_mean( w_mu * tf.square(y_p[:, -npar:] - y_d[:, -npar:] ))

        
    def mae_loc(predicted_y, desired_y):
        return tf.reduce_mean( tf.abs(predicted_y[:, :-num_parameters] - desired_y[:, :-num_parameters] ) ) 

    def mae_mu(predicted_y, desired_y):
        return tf.reduce_mean( tf.abs(predicted_y[:, -num_parameters:] - desired_y[:, -num_parameters:] ) ) 
    

    # model.compile(loss='mse',
    #             optimizer=optimizer,
    #             metrics=[mae_mu,mae_loc])
    
    
    model.compile(loss=custom_loss,
                optimizer=optimizer,
                metrics=[mae_mu,mae_loc])
    
    # Store training stats
    history = model.fit(X_train, y_train, epochs=EPOCHS,
                        validation_split=0.2, verbose=1,
                        callbacks=[PrintDot(),decay_lr], batch_size = 32)
    

    return model, history


def build_tensorflow_model_weights_justModel(N_in, N_out, num_parameters, network_width='normal', 
                                   lr = 1.e-4, decay = 1.e-2, w_l = 1.0, w_mu = 1.0):
    
    model = construct_tf_std_model(N_in, N_out, network_width)

    # optimizer= tf.train.AdamOptimizer(lr)
    optimizer= keras.optimizers.Adam(learning_rate = lr)
            
    npar = num_parameters
    
    def mae_loc(predicted_y, desired_y):
        return tf.reduce_mean( tf.abs(predicted_y[:, :-num_parameters] - desired_y[:, :-num_parameters] ) ) 

    def mae_mu(predicted_y, desired_y):
        return tf.reduce_mean( tf.abs(predicted_y[:, -num_parameters:] - desired_y[:, -num_parameters:] ) ) 
    

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=[mae_mu,mae_loc])
    
    
    # model.compile(loss=custom_loss,
    #             optimizer=optimizer,
    #             metrics=[mae_mu,mae_loc])
    
    return model

def construct_tf_pde_model( number_of_inputs, number_of_output, num_parameters, pde_activation, network_width='normal'):

    print( "number of parameters while constructing the network %d with network %s " % (num_parameters, network_width) )
    
    if network_width == 'normal':
        
        print( "PDE network with network normal " )

        pde_model = keras.Sequential([
        keras.layers.Dense( 1024, activation=tf.nn.relu,
                       input_shape=(number_of_inputs,)),
            keras.layers.Dense( 512, activation=tf.nn.relu),
            keras.layers.Dense( 256, activation=tf.nn.relu),
            keras.layers.Dense( 128, activation=tf.nn.relu),
            keras.layers.Dense( num_parameters, activation=tf.nn.sigmoid ),
            tf.keras.layers.Lambda(pde_activation.pde_mu_solver, \
                                 output_shape=( number_of_output, ) )
          ])
    elif network_width == 'small':
        
        print( "PDE network with network small " )

        pde_model = keras.Sequential([
            keras.layers.Dense( 64, activation=tf.nn.relu,
                       input_shape=(number_of_inputs,)),
            keras.layers.Dense( 64, activation=tf.nn.relu),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense( num_parameters, activation=tf.nn.sigmoid ),
            tf.keras.layers.Lambda(pde_activation.pde_mu_solver, \
                                 output_shape=( number_of_output, ) )
          ])
            
    elif network_width == 'constant':
        print( "TF network with network constant " )

        pde_model = keras.Sequential([
        keras.layers.Dense( 256, activation=tf.nn.relu,
                            input_shape=(number_of_inputs,)),
        keras.layers.Dense( 256, activation=tf.nn.relu),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense( num_parameters, activation=tf.nn.sigmoid ),
        tf.keras.layers.Lambda(pde_activation.pde_mu_solver, \
                                 output_shape=( number_of_output, ) )
        ])
    elif network_width == 'large':
        
        print( "PDE network with network large " )

        pde_model = keras.Sequential([
        keras.layers.Dense( 2048, activation=tf.nn.relu,
                       input_shape=(number_of_inputs,)),
            keras.layers.Dense( 2048, activation=tf.nn.relu),
            keras.layers.Dense(2048, activation=tf.nn.relu),
            keras.layers.Dense(2048, activation=tf.nn.relu),
            keras.layers.Dense( num_parameters, activation=tf.nn.sigmoid ),
            tf.keras.layers.Lambda(pde_activation.pde_mu_solver, \
                                 output_shape=( number_of_output, ) )
          ])
            
    elif network_width == 'modified_felipe':   
        Nneurons = 64
        print( "TF network with modified " )
        pde_model = keras.Sequential([
            keras.layers.Dense( Nneurons, activation=tf.keras.activations.linear,
                                input_shape=(number_of_inputs,)),
            keras.layers.Dense( Nneurons, activation=tf.nn.relu),
            keras.layers.Dense( Nneurons, activation=tf.nn.leaky_relu),
            keras.layers.Dense( Nneurons, activation=tf.nn.relu),
            keras.layers.Dense( Nneurons, activation=tf.nn.relu),
            keras.layers.Dense(num_parameters, activation=tf.nn.sigmoid ),
            tf.keras.layers.Lambda(pde_activation.pde_mu_solver, \
                                 output_shape=( number_of_output, ))
          ])

    elif network_width == 'modified_felipe_autoencoder':   
        Nneurons = 64
        print( "TF network with modified " )
        pde_model = keras.Sequential([
            keras.layers.Dense( Nneurons, activation=tf.keras.activations.linear,
                                input_shape=(number_of_inputs,)),
            keras.layers.Dense( Nneurons, activation=tf.nn.relu),
            keras.layers.Dense( Nneurons, activation=tf.nn.leaky_relu),
            keras.layers.Dense( Nneurons, activation=tf.nn.relu),
            keras.layers.Dense( Nneurons, activation=tf.nn.relu),
            keras.layers.Dense( Nneurons/2, activation=tf.nn.relu),
            keras.layers.Dense( Nneurons/4, activation=tf.nn.relu),
            keras.layers.Dense( Nneurons/8, activation=tf.nn.relu),
            keras.layers.Dense( Nneurons/16, activation=tf.nn.relu),
            keras.layers.Dense(num_parameters, activation=tf.nn.sigmoid ),
            tf.keras.layers.Lambda(pde_activation.pde_mu_solver, \
                                 output_shape=( number_of_output, ))])
    
    # NB for tf _computed_mu will gather ALL the parameters for all the training data!!
#    pde_model.add(tf.keras.layers.Lambda(pde_activation.pde_mu_solver, \
#                                 output_shape=( number_of_output, ) ) )
#                                 , input_shape=( 3, ) ) 

    return pde_model
    



def build_pde_tensorflow_model( X_train, y_train, EPOCHS, pde_activation, num_parameters, weight_mu, \
                                network_width='normal', optimizer='adam' , lr = 0.0001):

    number_of_output = y_train.shape[1]
    number_of_inputs = X_train.shape[1]

    model = construct_tf_pde_model( number_of_inputs, number_of_output, num_parameters, pde_activation, network_width, lr )


    # if optimizer=='gradient_descent':
    optimizer= tf.train.GradientDescentOptimizer( 0.01 )
    # elif optimizer == 'adam':
    #     optimizer = tf.train.AdamOptimizer(lr)

#    weight_mu  = 100.0  # float( number_of_output_locations ) / float( number_of_output )
    weight_loc = 1.0  # float( num_parameters ) / float( number_of_output )
    
    def custom_loss(predicted_y, desired_y):
        return tf.reduce_mean( weight_loc * tf.square(predicted_y[:, :-num_parameters] - desired_y[:, :-num_parameters] ) 
             + tf.reduce_mean( weight_mu  * tf.square(predicted_y[:, -num_parameters:] - desired_y[:, -num_parameters:] ) ) ) 
        
    # def mean_absolute_error(predicted_y, desired_y):
    #     return tf.reduce_mean( tf.abs(predicted_y[:, :-num_parameters] - desired_y[:, :-num_parameters] ) ) 

    def mae_loc(predicted_y, desired_y):
        return tf.reduce_mean( tf.abs(predicted_y[:, :-num_parameters] - desired_y[:, :-num_parameters] ) ) 

    def mae_mu(predicted_y, desired_y):
        return tf.reduce_mean( tf.abs(predicted_y[:, -num_parameters:] - desired_y[:, -num_parameters:] ) )                     
        
    model.compile(loss=custom_loss,
                  optimizer=optimizer,
                  metrics=[mae_mu,mae_loc] )
    batchsize = int(np.max([X_train.shape[0]/20,100]))

    # Store training stats
    history = model.fit(X_train, y_train, epochs=EPOCHS,
                        validation_split=0.2, verbose=1,
                        callbacks=[PrintDot()], batch_size = 32)

    return model, history

def build_pde_tensorflow_model_felipe( X_train, y_train, EPOCHS, pde_activation, num_parameters, weight_mu, \
                                network_width='normal', optimizer='adam' , lr = 1.e-4, decay = 1.e-2):

    number_of_output = y_train.shape[1]
    number_of_inputs = X_train.shape[1]

    model = build_pde_tensorflow_model_felipe_justModel( number_of_inputs, number_of_output , pde_activation, num_parameters, weight_mu, \
                                network_width, optimizer, lr)

    
    
    def scheduler(epoch):
        omega = np.sqrt(float(epoch/EPOCHS))
        rate = lr*(1.0 - omega) + omega*decay*lr
        print('learning_rate = ', rate)
        return rate
        
    decay_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
      

    # Store training stats
    history = model.fit(X_train, y_train, epochs=EPOCHS,
                        validation_split=0.2, verbose=1,
                        callbacks=[PrintDot(),decay_lr], batch_size = 32)

    return model, history



def build_pde_tensorflow_model_felipe_justModel( number_of_inputs, number_of_output , pde_activation, num_parameters, weight_mu, \
                                network_width='normal', optimizer='adam' , lr = 1.e-4):

    model = construct_tf_pde_model( number_of_inputs, number_of_output, num_parameters, pde_activation, network_width)

    optimizer= keras.optimizers.Adam(learning_rate = lr)

#    weight_mu  = 100.0  # float( number_of_output_locations ) / float( number_of_output )
    weight_loc = 1.0  # float( num_parameters ) / float( number_of_output )
    
    def custom_loss(predicted_y, desired_y):
        return tf.reduce_mean( weight_loc * tf.square(predicted_y[:, :-num_parameters] - desired_y[:, :-num_parameters] ) 
             + tf.reduce_mean( weight_mu  * tf.square(predicted_y[:, -num_parameters:] - desired_y[:, -num_parameters:] ) ) ) 

    def mae_loc(predicted_y, desired_y):
        return tf.reduce_mean( tf.abs(predicted_y[:, :-num_parameters] - desired_y[:, :-num_parameters] ) ) 

    def mae_mu(predicted_y, desired_y):
        return tf.reduce_mean( tf.abs(predicted_y[:, -num_parameters:] - desired_y[:, -num_parameters:] ) )                     
        
    model.compile(loss=custom_loss,
                  optimizer=optimizer,
                  metrics=[mae_mu,mae_loc] )

    return model




def evaluate_model( model, name, X_test, y_test, num_parameters, simulation_name="", folder="" ):

    test_predictions = model.predict(X_test)
    model_errors = abs( test_predictions - y_test )
    fem_output_coordinates = model_errors.shape[1] - num_parameters

    ns_test = y_test.shape[0]

    error_test = np.sqrt( np.sum( model_errors[:, 0:fem_output_coordinates] * 
                                  model_errors[:, 0:fem_output_coordinates] ) )
    
    error_test_local = np.zeros( ns_test )
    
    for ii in range( ns_test ):
        error_test_local[ii] = np.sqrt( np.sum( model_errors[ii, 0:fem_output_coordinates] * 
                                                model_errors[ii, 0:fem_output_coordinates] ) )
    
    
    error_test = np.mean( error_test_local )
    
    plt.figure( figsize=(15,5) )
    
    for iP in range ( num_parameters ):
        plt.scatter(  y_test[:,-num_parameters+iP], error_test_local, label='param_' + str(iP) )
    
    plt.title( name + ' true param vs error on u(x)' )
    plt.legend( )
    plt.grid( linestyle='--', linewidth=0.5 )
    
    print( "%s Error on test set is %f" % ( name, error_test )  )
    
    error_test_parameters = np.zeros( num_parameters )


    for iP in range( num_parameters ):
        error_test_parameters[iP] = np.mean( np.sqrt( model_errors[:, -num_parameters+iP] * model_errors[:, -num_parameters+iP] ) )
        print( "Error on parameter %d is %f " % ( iP, error_test_parameters[iP] ) )

    plt.figure( )

    f, (ax1, ax2) = plt.subplots( 1, 2, figsize=(15,5) )

    for iP in range( num_parameters ):
        ax1.scatter( y_test[:,-num_parameters+iP] , test_predictions[:,-num_parameters+iP], label='param_' + str(iP) )

    for iP in range( num_parameters ):
        ax2.scatter( y_test[:,-num_parameters+iP] , np.abs( test_predictions[:,-num_parameters+iP] - y_test[:,-num_parameters+iP]), label='param_' + str(iP) )

    ax1.set_title( name + ' results y_test vs y_pred' )
    ax2.set_title( name + ' errors  |y_pred-y_test| vs y_pred' )
    ax1.legend( )
    ax2.legend( )

    ax1.grid( linestyle='--', linewidth=0.5 )
    ax2.grid( linestyle='--', linewidth=0.5 )

    if folder != "" and folder[-1] != '/':
        folder = folder + '/'

    plt.savefig( folder + name + '_' + simulation_name + '.eps' )

    if simulation_name != "":
        output_file = open( folder + name + '_' + simulation_name + '.txt', 'w+' )
        
        for iP in range( num_parameters ):
            output_file.write( "%s-error parameter %d is %f \n" % (name, iP, error_test_parameters[iP]) )
    
        output_file.write( "%s-error on function value is on average %f \n" % (name, error_test ) )
        
        output_file.close( )
        
        output_file = open( folder + name + '_data_' + simulation_name + '.txt', 'w+' )
        
        for iP in range( num_parameters ):
            output_file.write( '%f\n' % (error_test_parameters[iP]) )
    
        output_file.write( '%f\n' % (error_test ) )
        
        output_file.close( )

    return error_test_parameters, error_test


def compare_two_model( model_1, model_2, name_1, name_2, X_test, y_test, num_parameters, config_1=-1, config_2=-1 ):
    
    test_predictions_1 = model_1.predict(X_test)

    plt.figure( )
    
    for iP in range( num_parameters ):
        plt.scatter( y_test[:,-(iP+1)] , np.abs(y_test[:,-(iP+1)] - test_predictions_1[:,-(iP+1)]), label='param ' + str(iP) + ' -- ' + name_1 )

    
    test_predictions_2 = model_2.predict(X_test)

    for iP in range( num_parameters ):
        plt.scatter( y_test[:,-(iP+1)] , np.abs(y_test[:,-(iP+1)] - test_predictions_2[:,-(iP+1)]), label='param ' + str(iP) + ' -- ' + name_2 )

    plt.title( 'Errors y_test vs y_pred' )
    plt.legend( )
    
#    plt.show( )
    
#    if config_1!= -1 and config_2 != -1:
#        matplotlib.pylab.savefig( 'foo_' + str(config_1) + '_' + str(config_2) + '.png' )
    
    plt.close( )
    
    return 

def compare_with_rb_model( ns_test, X_test, rb_manager, num_parameters, noise_magnitude, parameter_handler, \
                           fem_output_coordinates, pde_model, _used_param ):
    
    pde_test_pred  = pde_model.predict(  X_test )

    rb_error             = np.zeros( ns_test )
    rb_noised_error      = np.zeros( ns_test )
    pde_network_error    = np.zeros( ns_test )
    
    min_param = parameter_handler.get_min_parameters( )
    max_param = parameter_handler.get_max_parameters( )
    range_x = 0.5
    rescaled_range_param = ( max_param - min_param ) / range_x
    
    for iS in range( ns_test ):
        uh = rb_manager.get_test_snapshot( iS )
        
        real_param = rb_manager.get_test_parameter( iS )
    
#        noised_param = real_param * ( 1. + noise_magnitude * np.random.normal( 0, 1, np.shape( real_param ) ) )
        noised_param = real_param + noise_magnitude * rescaled_range_param * np.random.normal( 0, 1, np.shape( real_param ) )
        
        un = rb_manager.solve_reduced_problem( real_param, _used_Qa=_used_param )
        un_noised_param = rb_manager.solve_reduced_problem( noised_param, _used_Qa=_used_param )
        
        rb_manager.reconstruct_fem_solution( un )
        utildeh = rb_manager.get_utildeh( )
        error_h = uh - utildeh
    
        rb_manager.reconstruct_fem_solution( un_noised_param )
        utildeh_noised_param = rb_manager.get_utildeh( )
        error_h_noised_param = uh - utildeh_noised_param
    
        pde_network_error_h = pde_test_pred[ iS, :-num_parameters ] - uh[fem_output_coordinates]
        
        rb_error[iS]           = np.sqrt( np.sum( error_h[fem_output_coordinates] * error_h[fem_output_coordinates] ) )
        
        rb_noised_error[iS]    = np.sqrt( np.sum( error_h_noised_param[fem_output_coordinates] \
                                                * error_h_noised_param[fem_output_coordinates] ) )
    
        pde_network_error[iS]  = np.sqrt( np.sum( pde_network_error_h * pde_network_error_h ) ) 
    
        print( "RB error test %d are %f and %f, whereas PDE-network error is %f " % (iS, rb_error[iS], rb_noised_error[iS], pde_network_error[iS]) )
    
    plt.figure( )
    plt.scatter( np.arange( ns_test ), pde_network_error, label='PDE_network' )
    plt.scatter( np.arange( ns_test ), rb_error, label='pde_model_rb error' )
    plt.scatter( np.arange( ns_test ), rb_noised_error, label='rb_noised_error error' )
    plt.legend( )
    plt.yscale('log')
    plt.ylim(10.**-6., 10.**-1.)
#    plt.show( )
    
    rb_mean        = np.mean( rb_error )
    rb_noise_mean  = np.mean( rb_noised_error )
    pde_mean       = np.mean( pde_network_error )
    
    print( rb_mean )
    print( rb_noise_mean )
    print( pde_mean )
    
    np.argmax(pde_network_error)
    
    better_than_rb = float( sum(rb_error > pde_network_error) ) / float( ns_test)
    better_than_rb_noised = float( sum(rb_noised_error > pde_network_error) ) / float( ns_test)
    
    print( better_than_rb )
    print( better_than_rb_noised )

    return ( rb_mean, rb_noise_mean, pde_mean, better_than_rb, better_than_rb_noised )
