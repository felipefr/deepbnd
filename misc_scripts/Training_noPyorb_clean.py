#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:18:51 2019

@author: felipefr
"""

# First try of construction of a NN to predict finite element solution given some previously computed snapshots.
# Also I tried to use the RB basis (also computed offline) computed by myself (no pyorb, in buildingRB_noPyorb.py), for the RB special layer.
# We should note that this script is independent of feamat, since it is only used in the finite element solving stage.
# Consistent but not optimal results, i.e., The NN is able to minimise the cost functional up to 10-3, for both training and validation sets.

# Todo: Investigate why the error doesnt go lower, maybe 10-5 - 10-6. Why in some cases validation error is so higher than training error.
# Todo: RB layer have been programed for a first version but don't work properly. Too high errors. Some problem or bug in formulation.


#import matlab.engine
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import numpy as np
#from functools import partial
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from functools import partial, update_wrapper
import pickle

import elasticity_pde_activation as eact

K.set_floatx('float64')

def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


#tf.executing_eagerly()

# This is only used in the RB layer
def custom_loss(y_true, y_pred, w_u, w_mu, N_u, N_mu):
    return tf.reduce_mean(w_u * tf.square( y_true[:, :-N_mu] - y_pred[:, :-N_mu])) + \
           tf.reduce_mean(w_mu * tf.square( y_true[:, -N_mu:] - y_pred[:, -N_mu:]))
           
def custom_loss2(y_true, y_pred):
    return tf.reduce_mean(tf.square( y_true - y_pred))


def removeDirichletSelectedNodes(nodes,selection):   
    flagsBoundary = ~np.any(nodes[:,2:]==2, axis = 1)
    return selection[flagsBoundary[selection]]    
 
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 50 == 0: print('')
        print('.')
        
        
def getNeuralNet(Ninput,Noutput,sizeLayers,learning_rate, loss, metrics =['mse']):
    
    layers = [keras.layers.Dense(sizeLayers[0], activation=tf.nn.relu, input_shape=(Ninput,))]        
    
    for s in sizeLayers[1:]:
        layers.append(keras.layers.Dense(s, activation=tf.nn.relu))
    
    layers.append(keras.layers.Dense(Noutput, activation=tf.nn.relu))  
    model = keras.Sequential(layers)
    
    optimizer = tf.optimizers.Adam( learning_rate=learning_rate )
    
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
    return model

# Not using this at moment
#def getNeuralNetRB(Ninput,Ntheta,Noutput,sizeLayers,lambdaFunction, learning_rate, loss, metrics =['mse']):
#    
#    layers = [keras.layers.Dense(sizeLayers[0], activation=tf.nn.relu, input_shape=(Ninput,))]        
#    
#    for s in sizeLayers[1:]:
#        layers.append(keras.layers.Dense(s, activation=tf.nn.relu))
#    
#    layers.append(keras.layers.Dense(Ntheta, activation=tf.nn.sigmoid))  
#    
#    layers.append(tf.keras.layers.Lambda(lambdaFunction, output_shape=(Noutput,), input_shape=( Ntheta,)))
#    model = keras.Sequential(layers)
#    
#    optimizer = tf.optimizers.Adam( learning_rate=learning_rate )
#    
#    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
#    
#    return model


def sampleNodes(N,nodes,xmin,xmax,ymin,ymax):
    
    count = 0
    
    listI = []
    
    while len(listI)<N:
        
        i = np.random.randint(low = 0, high = Nnodes, size = 1)[0]
        
        x = nodes[i,0]
        y = nodes[i,1]
        
#        print('trying to include ', x , ' and ', y )
        if(x>xmin and x<xmax and y>ymin and y<ymax):
            listI.append(i)
#            print("point included", len(listI))
            if(len(listI)>N):
                listI = list(set(listI)) 
        else:
            pass
#            print('point not included')

    return np.array(listI,dtype = 'int')

# Loads mesh information
file = open("txt/nodes.txt",'rb') 
nodes = pickle.load(file)
file.close()

# Loads snapshots matrix and params, either for training/validation, and also for tests.
snapshots = np.loadtxt("txt/snapshots.txt")
snapshotsTest = np.loadtxt("txt/snapshotsTest.txt")
param = np.loadtxt("txt/param.txt")
paramTest = np.loadtxt("txt/paramTest.txt")

# Printing summary snapshots
ns = len(snapshots[0])
Nnodes = int(len(snapshots)/2)
print('Summary dataset')
print('ns=',ns)
print('Nnodes=',Nnodes)
print(np.argwhere(np.isnan(snapshots)))
print(np.argwhere(np.isnan(snapshotsTest)))


# Selecting input/output training/validation dataset. In this case it chooses randomly nodes to store solution. 
np.random.seed(2)
NfemInput = 30
NfemOutput = 30 # just in case of MLP_out
#RBsize = 0 # only used in RB layer

#inputCoord = np.unique(np.sort(np.random.randint(low = 0, high = Nnodes, size= NfemInput)))
#inputCoord = removeDirichletSelectedNodes(nodes,inputCoord)
inputCoord = sampleNodes(NfemInput,nodes,0.001,1.5,0.001,0.999)
inputDofs = np.concatenate((inputCoord, inputCoord + Nnodes))

X_train = snapshots[inputDofs,:].transpose()
X_test = snapshotsTest[inputDofs,:].transpose()

# Building the output (Uncomment just one case)
# i) case of MLP_out
#outputCoord = np.unique(np.sort(np.random.randint(low = 0, high = Nnodes, size= NfemOutput)))
#outputCoord = removeDirichletSelectedNodes(nodes,outputCoord)
outputCoord = sampleNodes(NfemOutput,nodes,1.5,2.999,0.001,0.999)
outputDofs = np.concatenate((outputCoord, outputCoord + Nnodes))
Y_train = snapshots[outputDofs,:].transpose()
Y_test = snapshotsTest[outputDofs,:].transpose()

# ii) case of MLP_mu
#Y_train = param
#Y_test = paramTest

# iii) case combined MLP_out + MLP_mu output 
#Y_train = np.concatenate( (snapshots[outputCoord,:],snapshots[outputCoord + Nnodes,:]), axis = 0 ).transpose()
#Y_test = np.concatenate( (snapshotsTest[outputCoord,:],snapshotsTest[outputCoord + Nnodes,:]), axis = 0 ).transpose()
#
#Y_train = np.concatenate((Y_train,param),axis = 1)
#Y_test = np.concatenate((Y_test,paramTest),axis = 1)


# Scaling using training as the standard
scalerX = MinMaxScaler()
scalerY = MinMaxScaler()

scalerX.fit(X_train)
scalerY.fit(Y_train)

X_train_norm = scalerX.transform(X_train)
Y_train_norm = scalerY.transform(Y_train)
X_test_norm = scalerX.transform(X_test)
Y_test_norm = scalerY.transform(Y_test)

# Network parameters
Ninput = len(X_train_norm[0])  
Noutput = len(Y_train_norm[0]) 

learning_rate = 0.01
use_rb_activation = False 
chosen_loss='mse'
batch_size = 10

EPOCHS = 300
validation_split = 0.2

my_metric = 'mse'

Nlayers = 10
NneuronsPerLayer = 128

#Test 1: 0.01, 10, 300, 0.2, 4, 256
#Test 2: 0.005, 10, 300, 0.2, 10, 256
#Test 3: 0.01, 10, 300, 0.2, 10, 256

# Customized metrics are not used at the moment. Maybe they are already good, but I need to check them.
#my_metric = wrapped_partial( custom_loss, w_u = 0.001, w_mu = 1.0, N_u = len(Y_train[0]) - 2, N_mu = 2) 
#wrapped_partial( custom_loss, w_u = 0.001, w_mu = 1.0, N_u = len(Y_train[0]) - 2, N_mu = 2) 
#my_metric_pureU = wrapped_partial( custom_loss, w_u = 1.0, w_mu = 0.0, N_u = len(Y_train[0]) - 2, N_mu = 2)
#my_metric_pureU.__name__ = 'my_metric_pureU'
#my_metric_pureMu = wrapped_partial( custom_loss, w_u = 0.0, w_mu = 1.0, N_u = len(Y_train[0]) - 2, N_mu = 2) 
#my_metric_pureMu.__name__ = 'my_metric_pureMu' 
#my_metric = custom_loss2


# Uncomment for Model MLP NN for output solution at nodes
model = getNeuralNet(Ninput,Noutput,Nlayers*[NneuronsPerLayer], learning_rate, my_metric, ['mse'])

# Uncomment for Model MLP NN + RB layer, for output as parameters
#rb = RBproblem(_N = 20 , _outputDofs = outputDofs)
#solveRBproblemPartial = rb.solveRBproblem
#model = getNeuralNetRB(Ninput,2,2,6*[64],solveRBproblemPartial, learning_rate, my_metric, ['mse'])



# Fitting the model (Uncomment one of the cases)
# i) case with shuffle and with builtin validation split
history = model.fit(x = X_train_norm, y=Y_train_norm, callbacks = [PrintDot()], epochs=EPOCHS, 
                    validation_split=0.2 , verbose=1, batch_size=batch_size)

# ii) case without shuffle and without builtin validation split
#history = model.fit(X_train_norm, Y_train_norm, callbacks = [PrintDot()], epochs=EPOCHS, validation_split=validation_split , verbose=1, batch_size=batch_size)


# iii) case validation set by myself, and with shuffle (? in this case)
#Nvalidation = int(validation_split * ns)
#validationSample = np.unique(np.random.randint(low = 0, high = ns, size = Nvalidation).astype('int'))
#trainingSample = np.array(list( set(np.arange(ns)) - set(validationSample)))
#history = model.fit(x = X_train_norm[trainingSample,:], y=Y_train_norm[trainingSample,:], callbacks = [PrintDot()], epochs=EPOCHS, 
#                    validation_data=(X_train_norm[validationSample,:], Y_train_norm[validationSample,:]) , verbose=1, batch_size=batch_size, shuffle = True)



# Evaluating prediction for the test set
Y_predict_norm = model.predict(X_test_norm)
model_errors = np.linalg.norm(Y_predict_norm - Y_test_norm, axis = 1)
error_test = np.mean(model_errors)

print("Error in test")
print(error_test)

# Now loss=mse = metrics, so just printing loss  
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.yscale('log')
plt.legend(['train','val'])
plt.yscale('log')
plt.grid()

plt.savefig("loss.png")

plt.show()

