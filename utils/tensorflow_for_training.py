import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np
import myTensorflow as mytf

import h5py
import pickle
import dataManipMisc as dman 
import myHDF5 as myhd
import symmetryLib as syml
import tensorflow as tf

ker = tf.keras.layers

def exportScale(filenameIn, filenameOut, nX, nY):
    scalerX, scalerY = syml.getDatasetsXY(nX, nY, filenameIn)[2:4]
    scalerLimits = np.zeros((max(nX,nY),4))
    scalerLimits[:nX,0] = scalerX.data_min_
    scalerLimits[:nX,1] = scalerX.data_max_
    scalerLimits[:nY,2] = scalerY.data_min_
    scalerLimits[:nY,3] = scalerY.data_max_

    np.savetxt(filenameOut, scalerLimits)

def importScale(filenameIn, nX, nY):
    scalerX = myMinMaxScaler()
    scalerY = myMinMaxScaler()
    scalerX.fit_limits(np.loadtxt(filenameIn)[:,0:2])
    scalerY.fit_limits(np.loadtxt(filenameIn)[:,2:4])
    scalerX.set_n(nX)
    scalerY.set_n(nY)
    
    return scalerX, scalerY

class myMinMaxScaler:
    def __init__(self):
        self.data_min_ = []
        self.data_max_ = []
        self.n = 0
        
    def set_n(self,n):
        self.n = n

        self.data_min_ = self.data_min_[:self.n]
        self.data_max_ = self.data_max_[:self.n]

    
    def fit_limits(self,limits):
        self.n = limits.shape[0]
                
        self.data_min_ = limits[:,0]
        self.data_max_ = limits[:,1]

        self.scaler = lambda x,i : (x - self.data_min_[i])/(self.data_max_[i]-self.data_min_[i])
        self.inv_scaler = lambda x,i : (self.data_max_[i]-self.data_min_[i])*x + self.data_min_[i]
                
    def fit(self,x):
        self.n = x.shape[1]
        
        for i in range(n):
            self.data_min_.append(x[:,i].min())
            self.data_max_.append(x[:,i].max())
        
        self.data_min_ = np.array(self.data_min_)
        self.data_max_ = np.array(self.data_max_)

        self.scaler = lambda x,i : (x - self.data_min_[i])/(self.data_max_[i]-self.data_min_[i])
        self.inv_scaler = lambda x,i : (self.data_max_[i]-self.data_min_[i])*x + self.data_min_[i]


    def transform(self,x):
        return np.array( [self.scaler(x[:,i],i) for i in range(self.n)] ).T
            
    def inverse_transform(self,x):
        return np.array( [self.inv_scaler(x[:,i],i) for i in range(self.n)] ).T

def simpleGeneralModel(Nin, Nout, net):
    Neurons, actLabels = net['Neurons'] , net['activations']

    x = [tf.keras.Input((Nin,))]
    
    for n,a in zip(Neurons + [Nout],actLabels): 
        x.append(ker.Dense(n, activation= mytf.dictActivations[a])(x[-1]))

    return tf.keras.Model(x[0], x[-1])


    # reg = tf.keras.regularizers.l1_l2(l1=0.1*lambReg, l2=lambReg)
    # ker.Dropout(drps[0])
    

def generalModel_dropReg(Nin, Nout, net):

    Neurons, actLabels, drps, lambReg = net['Neurons'] , net['activations'], net['drps'], net['reg']

    kw = {'kernel_constraint':tf.keras.constraints.MaxNorm(5.0), 
          'bias_constraint':tf.keras.constraints.MaxNorm(10.0), 
          'kernel_regularizer': tf.keras.regularizers.l1_l2(l1=0.1*lambReg, l2=lambReg), 
          'bias_regularizer': tf.keras.regularizers.l1_l2(l1=0.1*lambReg, l2=lambReg)} 
    
    x = [tf.keras.Input((Nin,))]
    
    x.append(ker.Dropout(drps[0])(x[0]))
    
    for n,a,d in zip(Neurons + [Nout],actLabels,drps[1:]): 
        x[1] = ker.Dense(n, activation= mytf.dictActivations[a],**kw)(x[1])
        x[1] = ker.Dropout(d)(x[1])
        
    return tf.keras.Model(x[0], x[1])
    
def basicModelTraining(XY_train, XY_val, model, net):
    
    Neurons = net['Neurons']
    actLabel = net['activations']
    Epochs = net['epochs']
    decay = net['decay']
    lr = net['lr']
    saveFile = net['file_weights'] 
    stepEpochs = net['stepEpochs']
    
    np.random.seed(1)
    
    Xtrain, Ytrain = XY_train
    indices = np.arange(len(Xtrain))
    np.random.shuffle(indices)

    Xtrain = Xtrain[indices,:]
    Ytrain = Ytrain[indices,:]

    w_l = (net['Y_data_max'] - net['Y_data_min'])**2.0  
    w_l = w_l.astype('float32')
    
    lossW= mytf.partial2(mytf.custom_loss_mse_2, weight = w_l)
        
    
    optimizer= tf.keras.optimizers.Adam(learning_rate = lr)
    model.compile(loss = lossW, optimizer=optimizer, metrics=[lossW,'mse','mae'])

    schdDecay = mytf.partial2(mytf.scheduler ,lr = lr, decay = decay, EPOCHS = Epochs)
    decay_lr = tf.keras.callbacks.LearningRateScheduler(schdDecay)    

    # decay_lr = tf.keras.callbacks.ReduceLROnPlateau(
    #     monitor='val_loss', factor=0.8, patience=10, verbose=1,
    #     mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

    
    kw = {}
    kw['epochs']= Epochs; kw['batch_size'] = 32
    kw['validation_data'] = XY_val
    kw['verbose'] = 1
    kw['callbacks']=[mytf.PrintDot(), decay_lr, mytf.checkpoint(saveFile,stepEpochs),
                      tf.keras.callbacks.CSVLogger(saveFile[:-5] + '_history.csv' , append=True)]
    
    history = model.fit(Xtrain, Ytrain, **kw)
    
    mytf.plot_history( history, label=['custom_loss_mse_2','val_custom_loss_mse_2'], savefile = saveFile[:-5] + '_plot')
        
    return history


def basicModelTraining_stress(XY_train, XY_val, model, net):
    
    Neurons = net['Neurons']
    actLabel = net['activations']
    Epochs = net['epochs']
    decay = net['decay']
    lr = net['lr']
    saveFile = net['file_weights'] 
    stepEpochs = net['stepEpochs']
    
    np.random.seed(1)
    
    Xtrain, Ytrain = XY_train
    indices = np.arange(len(Xtrain))
    np.random.shuffle(indices)

    Xtrain = Xtrain[indices,:]
    Ytrain = Ytrain[indices,:]

    fac = 1e2
    w_l = fac*(net['Y_data_max'] - net['Y_data_min'])**2.0  
    w_l[2] = 2.*w_l[2]
    w_l = w_l.astype('float32')
    
    lossW= mytf.partial2(mytf.custom_loss_mse_2, weight = w_l)
        
    
    optimizer= tf.keras.optimizers.Adam(learning_rate = lr)
    model.compile(loss = lossW, optimizer=optimizer, metrics=[lossW,'mse','mae'])

    schdDecay = mytf.partial2(mytf.scheduler ,lr = lr, decay = decay, EPOCHS = Epochs)
    decay_lr = tf.keras.callbacks.LearningRateScheduler(schdDecay)    

    # decay_lr = tf.keras.callbacks.ReduceLROnPlateau(
    #     monitor='val_loss', factor=0.8, patience=10, verbose=1,
    #     mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

    
    kw = {}
    kw['epochs']= Epochs; kw['batch_size'] = 32
    kw['validation_data'] = XY_val
    kw['verbose'] = 1
    kw['callbacks']=[mytf.PrintDot(), decay_lr, mytf.checkpoint(saveFile,stepEpochs),
                      tf.keras.callbacks.CSVLogger(saveFile[:-5] + '_history.csv' , append=True)]
    
    history = model.fit(Xtrain, Ytrain, **kw)
    
    mytf.plot_history( history, label=['custom_loss_mse_2','val_custom_loss_mse_2'], savefile = saveFile[:-5] + '_plot')
        
    return history

def writeDict(d):
    f = open(d['file_net'],'w')
    
    for keys, value in zip(d.keys(),d.values()):
        f.write("{0}: {1}\n".format(keys,value))
        
    f.close()