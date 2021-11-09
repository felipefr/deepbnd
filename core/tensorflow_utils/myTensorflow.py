from functools import partial, update_wrapper
import tensorflow as tf
# from tensorflow import tf.keras
# from tensorflow keras.optimizers import Adam
import numpy as np  
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint

dictActivations = {'tanh' : tf.nn.tanh, 
                    'sigmoid' : tf.nn.sigmoid , 
                    'linear': tf.keras.activations.linear,
                    'relu': tf.keras.activations.relu,
                    'leaky_relu': tf.nn.leaky_relu, 
                    'swish' : tf.keras.activations.swish}

# dictActivations = {'tanh' : tf.nn.tanh, 
#                    'sigmoid' : tf.nn.sigmoid , 
#                    'linear': tf.keras.activations.linear,
#                    'relu': tf.keras.activations.relu,
#                    'leaky_relu': tf.nn.leaky_relu}

dfInitK = tf.keras.initializers.glorot_uniform(seed = 1)
# dfInitK = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_in", distribution="untruncated_normal", seed=None)
dfInitB = tf.keras.initializers.Zeros()

# def partial2(func, *args, **kwargs):
#     partial_func = partial(func, *args, **kwargs)
#     update_wrapper(partial_func, func)
#     return partial_func

class CustomDense(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(CustomDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        return {"units": self.units}

# inputs = keras.Input((4,))
# outputs = CustomDense(10)(inputs)

# model = keras.Model(inputs, outputs)

class myMinMaxScaler:
    def __init__(self):
        self.data_min = []
        self.data_max = []
        self.n = 0
                
    def fit(self,x):
        self.n = x.shape[1]
        
        for i in range(n):
            self.data_min.append(x[:,i].min())
            self.data_max.append(x[:,i].max())
        
        self.data_min = np.array(self.data_min)
        self.data_max = np.array(self.data_max)

        self.scaler = lambda x,i : (x - self.data_min[i])/(self.data_max[i]-self.data_min[i])
        self.inv_scaler = lambda x,i : (self.data_max[i]-self.data_min[i])*x + self.data_min[i]


    def transform(self,x):
        return np.array( [self.scaler(x[:,i],i) for i in range(n)] ).T
            
    def inverse_transform(self,x):
        return np.array( [self.inv_scaler(x[:,i],i) for i in range(n)] ).T


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
        l1_reg = lambda w: lambReg * tf.linalg.norm(w,ord=1)
        
        reg = tf.keras.regularizers.l1_l2(l1=0.1*lambReg, l2=lambReg)
        
        self.nLayers = len(Nneurons)
        
        assert self.nLayers == len(drps) - 2 , "add dropout begin and end"
        
        self.ld = [tf.keras.layers.Dropout(drps[0])]
        self.la = [tf.keras.layers.Dense(Nneurons[0], activation=dictActivations[actLabel[0]], input_shape=input_shape, kernel_initializer=dfInitK, bias_initializer=dfInitB)]
            
        for n, drp in zip(Nneurons[1:],drps[1:-1]):
            self.ld.append(tf.keras.layers.Dropout(drp)) 
            self.la.append(tf.keras.layers.Dense(n, activation=dictActivations[actLabel[1]], 
                          kernel_regularizer=reg, bias_regularizer=reg, 
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

def DNNmodel_notFancy(n,m,Nneurons,activations):
    
    assert len(activations) == 3 
    
    L = len(Nneurons) - 1
    
    layers = [tf.keras.layers.Dense( Nneurons[0], activation=dictActivations[activations[0]], input_shape=(n,))]
    for i in range(L):
        layers.append(tf.keras.layers.Dense( Nneurons[i+1], activation=dictActivations[activations[1]]))
    layers.append(tf.keras.layers.Dense( m, activation=dictActivations[activations[2]])) 
    
    return tf.keras.Sequential(layers)


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
    
class DNNmodel_consensus(tf.keras.Model):
    def __init__(self, Nin, Nout, Neurons, actLabel, drps=None, lambReg=0.0):
        super(DNNmodel_consensus,self).__init__()
        
        self.net1 = DNNmodel(Nin, Nout, Neurons[0], actLabel[0], drps[0], lambReg[0])
        self.net2 = DNNmodel(Nin, Nout, Neurons[1], actLabel[1], drps[1], lambReg[1])
        
    def call(self, inputs):
        z1 = self.net1(inputs)
        z2 = self.net2(inputs)
        
        return 0.5*(z1 + z2)
    
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


def checkpoint(saveFile, stepEpochs = 1): 
    return ModelCheckpoint(saveFile, monitor='val_loss', verbose=1,
                           save_best_only=True, save_weights_only = True, mode='auto', period=stepEpochs)
    
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

def custom_loss_mse_2(y_p,y_d,weight):
    return tf.reduce_sum(tf.reduce_mean(tf.multiply(weight,tf.square(tf.subtract(y_p,y_d))), axis=0))

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




def plot_history(history,label=['loss','val_loss'], savefile=None, ):
  plt.figure(1)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.plot(history.epoch, np.array(history.history[label[0]]),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history[label[1]]),
           label = 'Val loss')
  plt.yscale('log')
  plt.legend()
#  plt.ylim([0, ])
  
  if savefile:
      plt.savefig(savefile + '.png')
      np.savetxt(savefile + '_history_.txt', np.array(history.history[label[0]]) )
      np.savetxt(savefile + '_history_val.txt', np.array(history.history[label[1]]) )
  
  plt.grid()    
  # plt.show( )

def my_train_model(model, X_train, y_train, num_parameters, EPOCHS , 
                                   lr = 1.e-4, decay = 1.e-2, w_l = 1.0, w_mu = 1.0, 
                                   ratio_val = 0.2, saveFile = 'save.hdf5', stepEpochs=1, validationSet = None):
    
    # NoisyAdam = add_gradient_noise(Adam)
    
    
    # optimizer= tf.keras.optimizers.Adam(learning_rate = lr, beta_1 = 0.87, beta_2=0.98) #beta_1=0.9, beta_2=0.999, epsilon=1e-07,
    optimizer= tf.keras.optimizers.Adam(learning_rate = lr)
    # optimizer= tf.keras.optimizers.RMSprop(learning_rate = lr)
    # optimizer= tf.keras.optimizers.Adadelta(learning_rate=lr)
    # optimizer= tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    
    # mseloss = lambda y_p, y_d : tf.reduce_mean( tf.square(tf.subtract(y_p, y_d) ))
    
    # myfactr = 1e1
    # optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss = mseloss, method = 'L-BFGS-B', 
    #                                                     options = {'maxiter': 1000,
    #                                                                 'maxfun': 50000,
    #                                                                 'maxcor': 70,
    #                                                                 'maxls': 70,
    #                                                                 # 'pgtol': myfactr * np.finfo(float).eps,
    #                                                                 # 'gtol': myfactr * np.finfo(float).eps,
    #                                                                 'disp': True})
    #                                                                 # 'ftol' : myfactr * np.finfo(float).eps})

    # with tf.Session() as session:
    #     optimizer.minimize(session)
  
    # model.compile(loss=custom_loss(w_l = w_l, w_mu = w_mu, npar = num_parameters),
    #             optimizer=optimizer,
    #             metrics=[mae_mu_(num_parameters), mae_loc_(num_parameters)])

    # model.compile(loss='mse', optimizer=optimizer, metrics = ['mse','mae'])
    lossW= partial2(custom_loss_mse_2, weight = w_l)
    model.compile(loss = lossW, optimizer=optimizer, metrics=[lossW,'mse','mae'])

    # model.compile(loss='mse',
    #             optimizer=optimizer,
    #             metrics=[partial2(mae_mu,npar = num_parameters), partial2(mae_loc , npar = num_parameters)])

    schdDecay = partial2(scheduler ,lr = lr, decay = decay, EPOCHS = EPOCHS)
    decay_lr = tf.keras.callbacks.LearningRateScheduler(schdDecay)    
    
    # Store training stats
    
    if(type(validationSet) == type(None)):
        print("training with a fixed split for the validation" )        
        history = model.fit(X_train, y_train, epochs=EPOCHS,
                            validation_split=ratio_val, verbose=1,
                            callbacks=[PrintDot(), decay_lr, checkpoint(saveFile,stepEpochs)], batch_size = 32)

    else:
        print("training with a given validation dataset" )
        history = model.fit(X_train, y_train, epochs=EPOCHS,
                    validation_data = (validationSet['X'],validationSet['Y']), verbose=1,
                    callbacks=[PrintDot(), decay_lr, checkpoint(saveFile,stepEpochs)], batch_size = 32)
        

    # Store training stats
    # history = model.fit(X_train, y_train, epochs=EPOCHS,
    #                     validation_split=0.2, verbose=1,
    #                     callbacks=[PrintDot()], batch_size = 32)
    

    return history

