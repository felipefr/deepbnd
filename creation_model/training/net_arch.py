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

        kw = {
               'kernel_constraint':tf.keras.constraints.MaxNorm(5.0), 
               'bias_constraint':tf.keras.constraints.MaxNorm(10.0), 
              'kernel_regularizer': tf.keras.regularizers.l2(lambReg), 
              'bias_regularizer': tf.keras.regularizers.l2(lambReg),
              'kernel_initializer': tf.keras.initializers.HeNormal()} 
        
        x = [tf.keras.Input((self.nX,))]
        
        x.append(tf.keras.layers.Dropout(drps[0])(x[0]))
        
        for n,a,d in zip(Neurons + [self.nY],actLabels,drps[1:]): 
            x[1] = tf.keras.layers.Dense(n, activation = mytf.dictActivations[a], **kw)(x[1])
            x[1] = tf.keras.layers.GaussianDropout(d)(x[1])
            
        return tf.keras.Model(x[0], x[1])
    
    
    def getModel_batchNorm(self):
        Neurons, actLabels, drps, lambReg = self.Neurons , self.activations, self.drps, self.reg

        kw = {
               'kernel_constraint':tf.keras.constraints.MaxNorm(5.0), 
               'bias_constraint':tf.keras.constraints.MaxNorm(10.0), 
              'kernel_regularizer': tf.keras.regularizers.l2(lambReg),
              'bias_regularizer': tf.keras.regularizers.l2(lambReg),
              'kernel_initializer': tf.keras.initializers.HeNormal()} 
        
        x = [tf.keras.Input((self.nX,))]
        
        for n,a,d in zip(Neurons + [self.nY],actLabels,drps[1:]): 
            if(len(x) == 1):
                x.append(tf.keras.layers.Dense(n, activation = mytf.dictActivations[a], **kw)(x[0]))
            else:
                x[1] = tf.keras.layers.GaussianNoise(stddev = 0.005)(x[1])
                x[1] = tf.keras.layers.Dense(n, activation = mytf.dictActivations[a], **kw)(x[1])
                x[1] = tf.keras.layers.GaussianDropout(rate = 0.005)(x[1])
            
        # regularizer = tf.keras.regularizers.OrthogonalRegularizer(factor=0.01)
        # layer = tf.keras.layers.Dense(units=4, kernel_regularizer=regularizer)

        return tf.keras.Model(x[0], x[1])
        

    def training(self, XY_train, XY_val, seed = 1):
        
        np.random.seed(seed)
      
        savefile = self.files['weights'] 
      
        Xtrain, Ytrain = XY_train
        indices = np.arange(len(Xtrain))
        np.random.shuffle(indices)
    
        Xtrain = Xtrain[indices,:]
        Ytrain = Ytrain[indices,:]
    
    
        if(isinstance(self.scalerY, dman.myMinMaxScaler)):
            w_l = (self.scalerY.data_max_ - self.scalerY.data_min_)**2.0  
       
        elif(isinstance(self.scalerY, dman.myNormalisationScaler)):
            w_l = (2.0*self.scalerY.data_std)**2.0
            
        w_l = w_l.astype('float32')
        
        lossW= mytf.my_partial(mytf.custom_loss_mse, weight = w_l)
            
        optimizer= tf.keras.optimizers.Adam(learning_rate = self.lr)
        
        model = self.getModel_batchNorm()
        model.summary()
        model.compile(loss = lossW, optimizer=optimizer, metrics=[lossW,'mse','mae'])
    
        schdDecay = mytf.my_partial(mytf.scheduler ,lr = self.lr, decay = self.decay, EPOCHS = self.epochs)
        decay_lr = tf.keras.callbacks.LearningRateScheduler(schdDecay)    
        
        kw = {}
        kw['epochs']= self.epochs; kw['batch_size'] = 32
        kw['validation_data'] = XY_val
        kw['verbose'] = 1
        kw['callbacks']=[decay_lr, mytf.checkpoint(savefile, self.stepEpochs, monitor = "val_custom_loss_mse" ),
                          tf.keras.callbacks.CSVLogger(savefile[:-5] + '_history.csv' , append=True)]
        
        history = model.fit(Xtrain, Ytrain, **kw)
        
        mytf.plot_history( history, label=['custom_loss_mse','val_custom_loss_mse'], savefile = savefile[:-5] + '_plot')
            
        return history


    def training_tensorboard(self, XY_train, XY_val, seed = 1):
        
        # reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_custom_loss_mse',
                                                        # factor=0.5, patience=2,
                                                        # min_delta=1e-6, cooldown = 0, min_lr = 1.0e-6 )
    
        np.random.seed(seed)
      
        savefile = self.files['weights'] 
      
        Xtrain, Ytrain = XY_train
        indices = np.arange(len(Xtrain))
        np.random.shuffle(indices)
    
        Xtrain = Xtrain[indices,:]
        Ytrain = Ytrain[indices,:]


        if(isinstance(self.scalerY, dman.myMinMaxScaler)):
            w_l = (self.scalerY.data_max_ - self.scalerY.data_min_)**2.0  
       
        elif(isinstance(self.scalerY, dman.myNormalisationScaler)):
            w_l = (2.0*self.scalerY.data_std)**2.0
            
        w_l = w_l.astype('float32')
        
        lossW= mytf.my_partial(mytf.custom_loss_mse, weight = w_l)
            
        optimizer= tf.keras.optimizers.Adam(learning_rate = self.lr)
        
        model = self.getModel_batchNorm()
        model.summary()
        model.compile(loss = lossW, optimizer=optimizer, metrics=[lossW,'mse','mae'])
    
        schdDecay = mytf.my_partial(mytf.scheduler_linear ,lr = self.lr, decay = self.decay, EPOCHS = self.epochs)
        decay_lr = tf.keras.callbacks.LearningRateScheduler(schdDecay)    

        
        kw = {}
        kw['epochs']= self.epochs; kw['batch_size'] = 32
        kw['validation_data'] = XY_val
        kw['verbose'] = 1
        kw['callbacks']=[mytf.PrintDot(), decay_lr, mytf.checkpoint(savefile, self.stepEpochs, monitor = "val_custom_loss_mse" ),
                          tf.keras.callbacks.CSVLogger(savefile[:-5] + '_history.csv' , append=True),
                          tensorboard_callback]
        
        
        
        history = model.fit(Xtrain, Ytrain, **kw)
        
        mytf.plot_history( history, label=['custom_loss_mse','val_custom_loss_mse'], savefile = savefile[:-5] + '_plot')
            
        return history
