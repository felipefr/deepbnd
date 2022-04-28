from functools import partial, update_wrapper
import tensorflow as tf
import numpy as np  
import matplotlib.pyplot as plt

import deepBND.creation_model.training.wrapper_tensorflow as mytf

import datetime
import deepBND.core.data_manipulation.utils as dman


class NetArch:
    
    def __init__(self, Neurons, activations, lr = 5.0e-4, decay = 1.0, drps = 0.0, reg = 0.0):
        self.Neurons = Neurons
        self.activations = activations
        self.lr = lr
        self.decay = decay
        self.drps = drps
        self.reg = reg
        
        self.epochs = None
        self.nY = None
        self.nX = None
        self.archId = None 
        self.stepEpochs = None
        self.scalerX = None
        self.scalerY = None
        self.files = {'weights' : None, 'net_settings': None, 'scaler': None,
                      'XY': None, 'XY_val': None}
    
    def getModel(self):
        Neurons, actLabels, drps, lambReg = self.Neurons , self.activations, self.drps, self.reg

        kw = {'kernel_constraint':tf.keras.constraints.MaxNorm(5.0), 
              'bias_constraint':tf.keras.constraints.MaxNorm(10.0), 
              'kernel_regularizer': tf.keras.regularizers.l1_l2(l1=0.1*lambReg, l2=lambReg), 
              'bias_regularizer': tf.keras.regularizers.l1_l2(l1=0.1*lambReg, l2=lambReg)} 
        
        x = [tf.keras.Input((self.nX,))]
        
        x.append(tf.keras.layers.Dropout(drps[0])(x[0]))
        
        for n,a,d in zip(Neurons + [self.nY],actLabels,drps[1:]): 
            x[1] = tf.keras.layers.Dense(n, activation = mytf.dictActivations[a], **kw)(x[1])
            x[1] = tf.keras.layers.Dropout(d)(x[1])
            
        return tf.keras.Model(x[0], x[1])
        

    def training(self, XY_train, XY_val, seed = 1):
        
        np.random.seed(seed)
      
        savefile = self.files['weights'] 
      
        Xtrain, Ytrain = XY_train
        indices = np.arange(len(Xtrain))
        np.random.shuffle(indices)
    
        Xtrain = Xtrain[indices,:]
        Ytrain = Ytrain[indices,:]
    
        w_l = (self.scalerY.data_max_ - self.scalerY.data_min_)**2.0  
        w_l = w_l.astype('float32')
        
        lossW= mytf.my_partial(mytf.custom_loss_mse, weight = w_l)
            
        optimizer= tf.keras.optimizers.Adam(learning_rate = self.lr)
        
        model = self.getModel()
        model.summary()
        model.compile(loss = lossW, optimizer=optimizer, metrics=[lossW,'mse','mae'])
    
        schdDecay = mytf.my_partial(mytf.scheduler ,lr = self.lr, decay = self.decay, EPOCHS = self.epochs)
        decay_lr = tf.keras.callbacks.LearningRateScheduler(schdDecay)    
        
        kw = {}
        kw['epochs']= self.epochs; kw['batch_size'] = 32
        kw['validation_data'] = XY_val
        kw['verbose'] = 1
        kw['callbacks']=[mytf.PrintDot(), decay_lr, mytf.checkpoint(savefile, self.stepEpochs),
                          tf.keras.callbacks.CSVLogger(savefile[:-5] + '_history.csv' , append=True)]
        
        history = model.fit(Xtrain, Ytrain, **kw)
        
        mytf.plot_history( history, label=['custom_loss_mse','val_custom_loss_mse'], savefile = savefile[:-5] + '_plot')
            
        return history


    def training_tensorboard(self, XY_train, XY_val, seed = 1):
        
        np.random.seed(seed)
      
        savefile = self.files['weights'] 
      
        Xtrain, Ytrain = XY_train
        indices = np.arange(len(Xtrain))
        np.random.shuffle(indices)
    
        Xtrain = Xtrain[indices,:]
        Ytrain = Ytrain[indices,:]
    
        if(type(self.scalerY) == type(dman.myMinMaxScaler())):
            w_l = (self.scalerY.data_max_ - self.scalerY.data_min_)**2.0  
       
        elif(type(self.scalerY) == type(dman.myNormalisationScaler()) ):
            w_l = (2.0*self.scalerY.data_std)**2.0
            
        w_l = w_l.astype('float32')
        
        lossW= mytf.my_partial(mytf.custom_loss_mse, weight = w_l)
            
        optimizer= tf.keras.optimizers.Adam(learning_rate = self.lr)
        
        model = self.getModel()
        model.summary()
        model.compile(loss = lossW, optimizer=optimizer, metrics=[lossW,'mse','mae'])
    
        schdDecay = mytf.my_partial(mytf.scheduler_linear ,lr = self.lr, decay = self.decay, EPOCHS = self.epochs)
        decay_lr = tf.keras.callbacks.LearningRateScheduler(schdDecay)    

                
        log_dir = "./" + self.files["tensorboard_id"]
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)   
        
        kw = {}
        kw['epochs']= self.epochs; kw['batch_size'] = 32
        kw['validation_data'] = XY_val
        kw['verbose'] = 1
        kw['callbacks']=[mytf.PrintDot(), decay_lr, mytf.checkpoint(savefile, self.stepEpochs),
                          tf.keras.callbacks.CSVLogger(savefile[:-5] + '_history.csv' , append=True),
                          tensorboard_callback]
        
        history = model.fit(Xtrain, Ytrain, **kw)
        
        mytf.plot_history( history, label=['custom_loss_mse','val_custom_loss_mse'], savefile = savefile[:-5] + '_plot')
            
        return history




standardNets = {'big': NetArch([300, 300, 300], 3*['swish'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.005] + [0.0], 1.0e-8),
         'small':NetArch([40, 40, 40], 3*['swish'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.005] + [0.0], 1.0e-8)}

#standardNets = {'big': NetArch([300, 300, 300], 3*['relu'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.005] + [0.0], 1.0e-8),
    #       'small': NetArch([40, 40, 40], 3*['relu'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.01] + [0.0], 1.0e-7)}
