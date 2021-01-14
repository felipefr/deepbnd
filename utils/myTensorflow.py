from functools import partial, update_wrapper
import tensorflow as tf
# from tensorflow import tf.keras
# from tensorflow keras.optimizers import Adam
import numpy as np  
import matplotlib.pyplot as plt

dictActivations = {'tanh' : tf.nn.tanh, 
                   'sigmoid' : tf.nn.sigmoid , 
                   'linear': tf.keras.activations.linear,
                   'relu': tf.keras.activations.relu,
                   'leaky_relu': tf.nn.leaky_relu}


dfInitK = tf.keras.initializers.glorot_uniform(seed = 1)
# dfInitK = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_in", distribution="untruncated_normal", seed=None)
dfInitB = tf.keras.initializers.Zeros()

def set_seed_default_initialisers(seed):
    return tf.keras.initializers.glorot_uniform(seed = 1), tf.keras.initializers.Zeros()
    # return tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_in", distribution="untruncated_normal", seed=seed), tf.keras.initializers.Zeros()


def mySequential(y,layers):
    
    for l in layers:
        y = l(y)
    
    return y


def mySequential2(y,layers1,layers2):
    
    for l1,l2 in zip(layers1,layers2):
        y = l1(y)
        y = l2(y)
    
    return y


class MLPblock(tf.keras.layers.Layer):
    def __init__(self, Nneurons, input_shape): 
        super(MLPblock, self).__init__()

      
        self.hidden_layers = [
        tf.keras.layers.Dense( Nneurons, activation=tf.keras.activations.linear,
                            input_shape=input_shape),
        tf.keras.layers.Dense( Nneurons, activation=tf.nn.relu),
        tf.keras.layers.Dense(Nneurons, activation=tf.nn.leaky_relu),
        tf.keras.layers.Dense(Nneurons, activation=tf.nn.relu),
        tf.keras.layers.Dense(Nneurons, activation=tf.nn.relu)]
        
    def call(self, input_features):
        return mySequential(input_features,self.hidden_layers)


class MLPblock_gen(tf.keras.layers.Layer):
    def __init__(self, Nneurons, input_shape, drps, lambReg, actLabel): 
        super(MLPblock_gen, self).__init__()
        
        l2_reg = lambda w: lambReg * tf.linalg.norm(w)**2.0
        
        self.nLayers = len(Nneurons)
        
        assert self.nLayers == len(drps) - 2 , "add dropout begin and end"
        
        self.ld = [tf.keras.layers.Dropout(drps[0])]
        self.la = [tf.keras.layers.Dense(Nneurons[0], activation=dictActivations[actLabel[0]], input_shape=input_shape, kernel_initializer=dfInitK, bias_initializer=dfInitB)]
        
       
        for n, drp in zip(Nneurons[1:],drps[1:-1]):
            self.ld.append(tf.keras.layers.Dropout(drp)) 
            self.la.append(tf.keras.layers.Dense(n, activation=dictActivations[actLabel[1]], kernel_regularizer=l2_reg, 
                          kernel_constraint=tf.keras.constraints.MaxNorm(300.0), kernel_initializer=dfInitK, bias_initializer=dfInitB))   
        
        self.ld.append(tf.keras.layers.Dropout(drps[-1])) 

    def call(self, input_features):
        return mySequential2(input_features,self.la, self.ld)


class RBlayer(tf.keras.layers.Layer):
    def __init__(self, num_parameters, pde_mu_solver, output_shape): 
        super(RBlayer, self).__init__()

        self.paramLayer = tf.keras.layers.Dense(num_parameters, activation=tf.nn.sigmoid)
        self.RBsolver = tf.keras.layers.Lambda(pde_mu_solver, output_shape= output_shape)

    def call(self, input_features):
        return mySequential(input_features,[self.paramLayer,self.RBsolver])
    
class PDEDNNmodel(tf.keras.Model):
    def __init__(self, _num_parameters, _pde_mu_solver, _Ninput, _Noutput):
        super(PDEDNNmodel,self).__init__()
        
        self.num_parameters = _num_parameters
        self.pde_mu_solver = _pde_mu_solver
        self.Ninput = _Ninput
        self.Noutput = _Noutput
        
        self.mlpblock = MLPblock(Nneurons=64 , input_shape=(self.Ninput,))
        self.rblayer = RBlayer(self.num_parameters, self.pde_mu_solver, (self.Noutput,))
              
    def call(self, inputs):
        
        mu = self.mlpblock(inputs)
        y = self.rblayer(mu)
        
        return y
        

class DNNmodel(tf.keras.Model):
    def __init__(self, Nin, Nout, Neurons, actLabel, drps=None, lambReg=0.0):
        super(DNNmodel,self).__init__()
        
        self.Nin = Nin
        self.Nout = Nout
        
        if(type(drps) == type(None)):
            self.mlpblock = MLPblock(Nneurons=Neurons, input_shape=(self.Nin,))
        else:   
            self.mlpblock = MLPblock_gen(Neurons, (self.Nin,), drps, lambReg, actLabel[0:2])
            
        self.outputLayer = tf.keras.layers.Dense( Nout, activation=dictActivations[actLabel[2]], kernel_initializer=dfInitK, bias_initializer=dfInitB)
        
    def call(self, inputs):
        
        z = self.mlpblock(inputs)
        y = self.outputLayer(z)
        
        return y
    
class DNNmodel_in(tf.keras.Model):
    def __init__(self, Nin, Nout, Neurons, actLabel, drps=None, lambReg=0.0):
        super(DNNmodel,self).__init__()
        
        self.Nin = Nin
        self.Nout = Nout
        
        if(type(drps) == type(None)):
            self.mlpblock = MLPblock(Nneurons=Neurons, input_shape=(self.Nin,))
        else:   
            self.mlpblock = MLPblock_gen(Neurons, (self.Nin,), drps, lambReg, actLabel[0:2])
            
        self.outputLayer = tf.keras.layers.Dense( Nout, activation=dictActivations[actLabel[2]])
        
    def call(self, inputs):
        
        z = self.mlpblock(inputs)
        y = self.outputLayer(z)
        
        return y


class EncoderModel(tf.keras.Model):
    def __init__(self, Nin, Nout, nLayers, actLabel, drps=None, lambReg=0.0):
        super(EncoderModel,self).__init__()
        
        self.Nin = Nin
        self.Nout = Nout
        self.nLayers = nLayers
        self.ratio = (Nout/Nin)**(1./nLayers)
        
        Neurons = [int(Nin*self.ratio**i) for i in range(nLayers)] # just hidden ones
        
        self.mlpblock = MLPblock_gen(Neurons, (self.Nin,), drps, lambReg, actLabel[0:2])
            
        self.outputLayer = tf.keras.layers.Dense( Nout, activation=dictActivations[actLabel[2]])
        
    def call(self, inputs):
        
        z = self.mlpblock(inputs)
        y = self.outputLayer(z)
        
        return y
    


# Display training progress by printing a single dot for each completed epoch
class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
         if epoch % 100 == 0: print('')
         print('.', end='')
    
def scheduler(epoch, decay, lr, EPOCHS):    
    omega = np.sqrt(float(epoch/EPOCHS))
    rate = lr*(1.0 - omega) + omega*decay*lr
    print('learning_rate = ', rate)
    return rate
    
def custom_loss(w_l, w_mu, npar):
    return lambda y_p, y_d : tf.reduce_mean( w_l * tf.square(tf.subtract(y_p[:, :-npar],y_d[:, :-npar]) )) + tf.reduce_mean( w_mu * tf.square(tf.subtract(y_p[:, -npar:], y_d[:, -npar:]) ))
  
# def custom_loss(w_l, w_mu, npar):
#     return lambda y_p, y_d : tf.reduce_mean( w_l * tf.square(tf.subtract(y_p[:, :-npar],y_d[:, :-npar]) )) + tf.reduce_mean( w_mu * tf.square(tf.subtract(y_p[:, -npar:], y_d[:, -npar:]) ))

# def custom_loss_mse(weight):
#     return lambda y_p, y_d : tf.keras.losses.MeanSquaredError(y_p , y_d, sample_weight = weight)

def custom_loss_mse(weight):
    return lambda y_p, y_d : tf.reduce_mean(tf.multiply(weight,tf.reduce_sum(tf.square(tf.subtract(y_p,y_d)),axis = 0)))

def mae_loc_(npar):
    def mae_loc(y_p, y_d):
        return tf.reduce_mean( tf.abs(tf.subtract(y_p[:, :-npar], y_d[:, :-npar]) ) ) 
    
    return mae_loc    

def mae_mu_(npar):
    def mae_mu(y_p, y_d): 
        return tf.reduce_mean( tf.abs(tf.subtract(y_p[:, -npar:], y_d[:, -npar:]) ) ) 
    
    return mae_mu
    
def partial2(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

def construct_pdednn_model_deprecated( number_of_inputs, number_of_output, num_parameters, pde_activation, network_width='normal'):
    
    model = construct_model_hidden(number_of_inputs, network_width)
    
    model.add(tf.keras.layers.Dense(num_parameters, activation=tf.nn.sigmoid ))
    model.add(tf.keras.layers.Lambda(pde_activation.pde_mu_solver, output_shape=( number_of_output, )))
    
    return model

def construct_pdednn_model( number_of_inputs, number_of_output, num_parameters, pde_activation):
    
    model = tf.keras.Sequential()
    model.add(MLPblock(Nneurons=64 , input_shape=(number_of_inputs,)))
    model.add(RBlayer(num_parameters, pde_activation.pde_mu_solver, (number_of_output,)))
    
    return model
    
def construct_dnn_model(  number_of_inputs, number_of_output, network_width='normal', act = 'linear'):
    model = construct_model_hidden(number_of_inputs, network_width)
    
    if(act == 'sigmoid'):
        model.add( tf.keras.layers.Dense( number_of_output, activation=tf.nn.sigmoid ) )
    elif(act == 'linear'):
        model.add( tf.keras.layers.Dense( number_of_output, activation=tf.keras.activations.linear))

    return model

def construct_model_hidden( number_of_inputs, network_width='normal'):
    

    print( "TF network with network %s " % network_width )
    
    if network_width == 'normal':
        
        print( "TF network with network normal " )

        model = tf.keras.Sequential([
        tf.keras.layers.Dense( 1024, activation=tf.nn.relu,
                           input_shape=(number_of_inputs,)),
            tf.keras.layers.Dense( 512, activation=tf.nn.relu),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu)
      ])
    
    elif network_width == 'small':
        
        print( "TF network with network small " )

        model = tf.keras.Sequential([
        tf.keras.layers.Dense( 64, activation=tf.nn.relu,
                            input_shape=(number_of_inputs,)),
        tf.keras.layers.Dense( 64, activation=tf.nn.relu),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(64, activation=tf.nn.relu)
      ])
        
    elif network_width == 'constant':
        print( "TF network with network constant " )

        model = tf.keras.Sequential([
        tf.keras.layers.Dense( 256, activation=tf.nn.relu,
                            input_shape=(number_of_inputs,)),
        tf.keras.layers.Dense( 256, activation=tf.nn.relu),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dense(256, activation=tf.nn.relu)
      ])
    
    elif network_width == 'large':
        
        print( "TF network with network large " )

        model = tf.keras.Sequential([
                tf.keras.layers.Dense( 2048, activation=tf.nn.relu,
                           input_shape=(number_of_inputs,)),
            tf.keras.layers.Dense( 2048, activation=tf.nn.relu),
            tf.keras.layers.Dense(2048, activation=tf.nn.relu),
            tf.keras.layers.Dense(2048, activation=tf.nn.relu)
      ])
    elif network_width == 'modified_felipe':   
        Nneurons = 64
        print( "TF network with modified " )
        model = tf.keras.Sequential([
            tf.keras.layers.Dense( Nneurons, activation=tf.keras.activations.linear,
                                input_shape=(number_of_inputs,)),
            tf.keras.layers.Dense( Nneurons, activation=tf.nn.relu),
            tf.keras.layers.Dense(Nneurons, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(Nneurons, activation=tf.nn.relu),
            tf.keras.layers.Dense(Nneurons, activation=tf.nn.relu)
            ])

    elif network_width == 'modified_felipe_autoencoder':   
        Nneurons = 64
        print( "TF network with modified " )
        model = tf.keras.Sequential([
            tf.keras.layers.Dense( Nneurons, activation=tf.keras.activations.linear,
                                input_shape=(number_of_inputs,)),
            tf.keras.layers.Dense( Nneurons, activation=tf.nn.relu),
            tf.keras.layers.Dense( Nneurons, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense( Nneurons, activation=tf.nn.relu),
            tf.keras.layers.Dense( Nneurons, activation=tf.nn.relu),
            tf.keras.layers.Dense( Nneurons/2, activation=tf.nn.relu),
            tf.keras.layers.Dense( Nneurons/4, activation=tf.nn.relu),
            tf.keras.layers.Dense( Nneurons/8, activation=tf.nn.relu)])

    elif network_width == 'modified_felipe2':   
        model = tf.keras.Sequential(MLPblock(Nneurons=256 , input_shape=(number_of_inputs,)))

    return model




def plot_history(history,savefile=None):
  plt.figure(1)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.plot(history.epoch, np.array(history.history['loss']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_loss']),
           label = 'Val loss')
  plt.yscale('log')
  plt.legend()
#  plt.ylim([0, ])
  
  if savefile:
      plt.savefig(savefile + '.png')
      np.savetxt(savefile + '_history_.txt', np.array(history.history['loss']) )
  
  plt.grid()    
  # plt.show( )

def my_train_model(model, X_train, y_train, num_parameters, EPOCHS , 
                                   lr = 1.e-4, decay = 1.e-2, w_l = 1.0, w_mu = 1.0, ratio_val = 0.2):
    
    # NoisyAdam = add_gradient_noise(Adam)
    
    
    # optimizer= tf.keras.optimizers.Adam(learning_rate = lr, beta_1 = 0.9)
    optimizer= tf.keras.optimizers.Adam(learning_rate = lr)
    # optimizer= tf.keras.optimizers.RMSprop(learning_rate = lr)
    # optimizer= tf.keras.optimizers.Adadelta(learning_rate=1.0)
    # optimizer= tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    
    # mseloss = lambda y_p, y_d : tf.reduce_mean( tf.square(tf.subtract(y_p, y_d) ))
    
    # myfactr = 1e1
    # optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss = mseloss, method = 'L-BFGS-B', 
    #                                                     options = {'maxiter': 1000,
    #                                                                'maxfun': 50000,
    #                                                                'maxcor': 70,
    #                                                                'maxls': 70,
    #                                                                # 'pgtol': myfactr * np.finfo(float).eps,
    #                                                                # 'gtol': myfactr * np.finfo(float).eps,
    #                                                                'disp': True})
    #                                                                # 'ftol' : myfactr * np.finfo(float).eps})

    # with tf.Session() as session:
    #     optimizer.minimize(session)
  
    # model.compile(loss=custom_loss(w_l = w_l, w_mu = w_mu, npar = num_parameters),
    #             optimizer=optimizer,
    #             metrics=[mae_mu_(num_parameters), mae_loc_(num_parameters)])

    # model.compile(loss='mse', optimizer=optimizer, metrics = ['mse','mae'])
    model.compile(loss = custom_loss_mse(weight = w_l),
                optimizer=optimizer,
                metrics=[custom_loss_mse(weight = w_l),'mse','mae'])

    # model.compile(loss='mse',
    #             optimizer=optimizer,
    #             metrics=[partial2(mae_mu,npar = num_parameters), partial2(mae_loc , npar = num_parameters)])

    decay_lr = tf.keras.callbacks.LearningRateScheduler(partial2(scheduler ,lr = lr, decay = decay, EPOCHS = EPOCHS))    
    
    # Store training stats
    history = model.fit(X_train, y_train, epochs=EPOCHS,
                        validation_split=ratio_val, verbose=1,
                        callbacks=[PrintDot(), decay_lr ], batch_size = 32)

    # Store training stats
    # history = model.fit(X_train, y_train, epochs=EPOCHS,
    #                     validation_split=0.2, verbose=1,
    #                     callbacks=[PrintDot()], batch_size = 32)
    

    return history

