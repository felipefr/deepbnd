from functools import partial, update_wrapper
import tensorflow as tf
import numpy as np  
import matplotlib.pyplot as plt

import deepBND.creation_model.training.wrapper_tensorflow as mytf

class NetArch:
    
    def __init__(self, Neurons, activations, lr = 5.0e-4, decay = 1.0, drps = 0.0, reg = 0.0):
        self.Neurons = Neurons
        self.activations = activations
        self.lr = lr
        self.decay = decay
        self.drps = drps
        self.reg = reg
        self.param = {}

    def getDict(self):
        d = {'Neurons': self.Neurons, 'activations': self.activations,
             'lr': self.lr, 'decay' : self.decay, 'drps' : self.drps, 'reg' : self.reg}
        
        d.update(self.param)
        
        return d
    
    def getModel(self, Nin, Nout):
        Neurons, actLabels, drps, lambReg = self.Neurons , self.activations, self.drps, self.reg

        kw = {'kernel_constraint':tf.keras.constraints.MaxNorm(5.0), 
              'bias_constraint':tf.keras.constraints.MaxNorm(10.0), 
              'kernel_regularizer': tf.keras.regularizers.l1_l2(l1=0.1*lambReg, l2=lambReg), 
              'bias_regularizer': tf.keras.regularizers.l1_l2(l1=0.1*lambReg, l2=lambReg)} 
        
        x = [tf.keras.Input((Nin,))]
        
        x.append(tf.keras.layers.Dropout(drps[0])(x[0]))
        
        for n,a,d in zip(Neurons + [Nout],actLabels,drps[1:]): 
            x[1] = tf.keras.layers.Dense(n, activation= mytf.dictActivations[a],**kw)(x[1])
            x[1] = tf.keras.layers.Dropout(d)(x[1])
            
        return tf.keras.Model(x[0], x[1])
        

    def training(self, XY_train, XY_val):
        
        dnet = self.getDict()
        Neurons = dnet['Neurons']
        actLabel = dnet['activations']
        Epochs = dnet['epochs']
        decay = dnet['decay']
        lr = dnet['lr']
        saveFile = dnet['file_weights'] 
        stepEpochs = dnet['stepEpochs']
        
        np.random.seed(1)
        
        Xtrain, Ytrain = XY_train
        indices = np.arange(len(Xtrain))
        np.random.shuffle(indices)
    
        Xtrain = Xtrain[indices,:]
        Ytrain = Ytrain[indices,:]
    
        w_l = (self.param['Y_data_max'] - self.param['Y_data_min'])**2.0  
        w_l = w_l.astype('float32')
        
        lossW= mytf.my_partial(mytf.custom_loss_mse, weight = w_l)
            
        optimizer= tf.keras.optimizers.Adam(learning_rate = lr)
        
        model = self.getModel(self.param['nX'], self.param['nY'])
        model.summary()
        model.compile(loss = lossW, optimizer=optimizer, metrics=[lossW,'mse','mae'])
    
        schdDecay = mytf.my_partial(mytf.scheduler ,lr = lr, decay = decay, EPOCHS = Epochs)
        decay_lr = tf.keras.callbacks.LearningRateScheduler(schdDecay)    
        
        kw = {}
        kw['epochs']= Epochs; kw['batch_size'] = 32
        kw['validation_data'] = XY_val
        kw['verbose'] = 1
        kw['callbacks']=[mytf.PrintDot(), decay_lr, mytf.checkpoint(saveFile,stepEpochs),
                          tf.keras.callbacks.CSVLogger(saveFile[:-5] + '_history.csv' , append=True)]
        
        history = model.fit(Xtrain, Ytrain, **kw)
        
        mytf.plot_history( history, label=['custom_loss_mse','val_custom_loss_mse'], savefile = saveFile[:-5] + '_plot')
            
        return history


nets = {'big': NetArch([300, 300, 300], 3*['swish'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.005] + [0.0], 1.0e-8),
         'small':NetArch([40, 40, 40], 3*['swish'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.005] + [0.0], 1.0e-8)}

#standardNets = {'big': NetArch([300, 300, 300], 3*['relu'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.005] + [0.0], 1.0e-8),
    #       'small': NetArch([40, 40, 40], 3*['relu'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.01] + [0.0], 1.0e-7)}
