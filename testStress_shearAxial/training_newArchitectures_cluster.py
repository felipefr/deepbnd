import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0,'../utils/')
import matplotlib.pyplot as plt
import numpy as np

import myTensorflow as mytf
from timeit import default_timer as timer

import h5py
import pickle
# import Generator as gene
import myHDF5 
import dataManipMisc as dman 
from sklearn.preprocessing import MinMaxScaler
import miscPosprocessingTools as mpt
import myHDF5 as myhd
import symmetryLib as syml
import tensorflow as tf

import json
import copy

ker = tf.keras.layers

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

    maxAlpha = scalerY.data_max_
    minAlpha = scalerY.data_min_
    
    print(scalerY.data_max_[0:5])
    print(scalerY.data_min_[0:5])
    
    w_l = (net['Y_data_max'] - net['Y_data_min'])**2.0  
    w_l = w_l.astype('float32')
    
    lossW= mytf.partial2(mytf.custom_loss_mse_2, weight = w_l)
        
    
    optimizer= tf.keras.optimizers.Adam(learning_rate = lr)
    model.compile(loss = lossW, optimizer=optimizer, metrics=[lossW,'mse','mae'])

    schdDecay = mytf.partial2(mytf.scheduler ,lr = lr, decay = decay, EPOCHS = Epochs)
    decay_lr = tf.keras.callbacks.LearningRateScheduler(schdDecay)    
    
    kw = {}
    kw['epochs']= Epochs; kw['batch_size'] = 32
    kw['validation_data'] = XY_val
    kw['verbose'] = 1
    kw['callbacks']=[mytf.PrintDot(), decay_lr, mytf.checkpoint(saveFile,stepEpochs), tf.keras.callbacks.CSVLogger(saveFile[:-5] + '_history.csv' , append=True)]
    
    history = model.fit(Xtrain, Ytrain, **kw)
    
    mytf.plot_history(history, label=['custom_loss_mse_2','val_custom_loss_mse_2'], savefile = saveFile[:-5] + '_plot')
    
    return history

def writeDict(d):
    f = open(d['file_net'],'w')
    
    for keys, value in zip(d.keys(),d.values()):
        f.write("{0}: {1}\n".format(keys,value))
        
    f.close()

folder = './models/dataset_shear1/dataset_new4/'
nameXY = folder +  'XY.h5'

folderVal = './models/dataset_shear3/'
nameXY_val = folderVal +  'XY_Wbasis4.h5'


# Nrb = int(sys.argv[1])
# epochs = int(sys.argv[2])
# archId = int(sys.argv[3])


Nrb = int(input('Nrb='))
epochs = int(input('epochs='))
archId = int(input('archId='))


nX = 36

print('Nrb is ', Nrb, 'epochs ', epochs)

# net = {'Neurons': [40], 'activations': 3*['swish'], 'lr': 1.0e-3, 'decay' : 1.0,
#         'drps' : 3*[0.0], 'reg' : 0.0}

# net = {'Neurons': [1000,1000,1000], 'activations': 4*['swish'], 'lr': 5.0e-4, 'decay' : 1.0, 'drps' : [0.0] + 3*[0.005] + [0.0], 'reg' : 1.0e-8}

# net = {'Neurons': [1000, 1000, 1000], 'activations': 3*['swish'] + ['linear'], 'lr': 5.0e-4, 'decay' : 0.1, 'drps' : [0.0] + 3*[0.005] + [0.0], 'reg' : 1.0e-8}
# net = {'Neurons': [100, 100, 100], 'activations': 3*['swish'] + ['linear'], 'lr': 5.0e-4, 'decay' : 0.1, 'drps' : [0.0] + 3*[0.01] + [0.0], 'reg' : 1.0e-7}
net = {'Neurons': [40], 'activations': 1*['swish'] + ['linear'], 'lr': 5.0e-4, 'decay' : 0.1, 'drps' : 3*[0.0], 'reg' : 0.0}



net['epochs'] = int(epochs)
net['nY'] = Nrb
net['nX'] = nX
net['archId'] = archId
net['nsTrain'] = int(10240) 
net['nsVal'] = int(5120)
net['stepEpochs'] = 1
net['file_weights'] = './models/dataset_shear3/models/weights_ny{0}_arch{1}.hdf5'.format(Nrb,archId)
net['file_net'] = './models/dataset_shear3/models/net_ny{0}_arch{1}.txt'.format(Nrb,archId)
net['file_prediction'] = './models/dataset_shear3/models/prediction_ny{0}_arch{1}.txt'.format(Nrb,archId)
net['file_XY'] = [nameXY, nameXY_val]

scalerX, scalerY = syml.getDatasetsXY(nX, Nrb, net['file_XY'][0])[2:4]

net['Y_data_max'] = scalerY.data_max_ 
net['Y_data_min'] = scalerY.data_min_
net['X_data_max'] = scalerX.data_max_
net['X_data_min'] = scalerX.data_min_
net['scalerX'] = scalerX
net['scalerY'] = scalerY
net['routine'] = 'generalModel_dropReg'

writeDict(net)

XY_train = syml.getDatasetsXY(nX, Nrb, net['file_XY'][0], scalerX, scalerY)[0:2]
XY_val = syml.getDatasetsXY(nX, Nrb, net['file_XY'][1], scalerX, scalerY)[0:2]


model = generalModel_dropReg(nX, Nrb, net)    
model.summary()


start = timer()
hist = basicModelTraining(XY_train, XY_val, model, net)
end = timer()

# oldWeights = './models/newArchitectures/new4/weights_ny{0}_arch{1}_retaken6.hdf5'.format(Nrb,archId)
# model.load_weights(oldWeights)


# Prediction 
# X, Y = syml.getDatasetsXY(nX, Nrb, net['file_XY'], scalerX, scalerY)[0:2]
# w_l = (scalerY.data_max_ - scalerY.data_min_)**2.0

# nsTrain = net['nsTrain']
# X_scaled = []; Y_scaled = []
# X_scaled.append(X[:nsTrain,:]); Y_scaled.append(Y[:nsTrain,:])
# X_scaled.append(X[nsTrain:,:]); Y_scaled.append(Y[nsTrain:,:])


# nameXYlist = ['./models/dataset_new4/XY.h5','./models/dataset_newTest2/XY_Wbasis4.h5','./models/dataset_test/XY_Wbasis4.h5']

# for nameXY in nameXYlist:
#     Xtemp, Ytemp = syml.getDatasetsXY(nX, Nrb, nameXY, scalerX, scalerY)[0:2]
#     X_scaled.append(Xtemp); Y_scaled.append(Ytemp)

# netp = net
# modelp = generalModel_dropReg(nX, Nrb, netp)   
# oldWeights = netp['file_weights']

# modelp.load_weights(oldWeights)

# error_stats = []
# for Xi, Yi in zip(X_scaled,Y_scaled): 
#     Yi_p = modelp.predict(Xi)
#     error = tf.reduce_sum(tf.multiply(w_l,tf.square(tf.subtract(Yi_p,Yi))), axis=1).numpy()
    
#     error_stats.append(np.array([np.mean(error),np.std(error), np.max(error), np.min(error)]))

# np.savetxt(net['file_prediction'],np.array(error_stats))
