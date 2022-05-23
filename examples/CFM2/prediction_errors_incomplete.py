#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 16:54:42 2022

@author: felipe
"""

import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np

from deepBND.__init__ import * 
import tensorflow as tf
import deepBND.creation_model.training.wrapper_tensorflow as mytf
# from deepBND.creation_model.training.net_arch import standardNets
from deepBND.creation_model.training.net_arch import NetArch
import deepBND.core.data_manipulation.utils as dman
import deepBND.core.data_manipulation.wrapper_h5py as myhd


standardNets = {'huge': NetArch([500, 500, 500], 3*['swish'] + ['linear'], 5.0e-4, 0.1, 5*[0.0], 1.0e-9),
                'escalonated': NetArch([50, 300, 500], 3*['swish'] + ['linear'], 5.0e-4, 0.1, 5*[0.0], 1.0e-9),
                'big': NetArch([300, 300, 300], 3*['swish'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.005] + [0.0], 1.0e-8)}

 
def compute_DNN_error(net, Ylabel, Xmask):

    scalerX, scalerY = dman.importScale(net.files['scaler'], nX, Nrb, scalerType = 'MinMax') # wrongly, but the network was scaled wrongly
    
    Xbar, Ybar = dman.getDatasetsXY(nX, Nrb, net.files['XY_test'], scalerX, scalerY, Ylabel = Ylabel)[0:2]
    
    Y = scalerY.inverse_transform(Ybar)
    
    model = net.getModel()   
    
    model.load_weights(net.files['weights'])
    
    
    if(type(Xmask) != type(None)):
        print("hello")
        lacking = np.array(list(set(list(np.arange(36))) - set(list(Xmask))))  
        for i in lacking:
            Xbar[:, i] = 0.5*(np.max(Xbar[:,i]) + np.min(Xbar[:,i]))
      
    
    Yp = scalerY.inverse_transform(model.predict(Xbar)) 
    error = tf.reduce_sum(tf.square(tf.subtract(Yp,Y)), axis=1).numpy()
        
    error_stats = np.array([np.mean(error),np.std(error), np.max(error), np.min(error)])

    return error, error_stats, error_stats[0]


def compute_POD_error(net, loadType):
    ns = len(myhd.loadhd5(net.files['XY_train'], 'X'))    
    eig = myhd.loadhd5(net.files['Wbasis'], 'sig_%s'%loadType)   
        
    errorPOD = np.sum(eig[net.nY:])/ns
    
    return errorPOD


def plot_curve_training(net, title = '', add_y_lines=[]):
    historic_file = net.files["weights"][:-5] + "_history.csv"
    
    hist = np.loadtxt(historic_file,skiprows=1,delimiter= ',')
    keys = list(np.loadtxt(historic_file,delimiter= ',', max_rows=1, dtype = type('s')))
    
    index_epochs = keys.index("epoch")                     
    index_weighted_loss = keys.index("custom_loss_mse")                 
    index_weighted_val_loss = keys.index("val_custom_loss_mse")
    
    plt.figure(1,(5.0,3.5))
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(hist[:,index_epochs] , hist[:,index_weighted_loss],label= 'Training Loss')
    plt.plot(hist[:,index_epochs] , hist[:,index_weighted_val_loss],label = 'Validation Loss')
    for yi in add_y_lines:
        plt.plot([0,np.max(hist[:,index_epochs])] , 2*[yi[0]] ,label= yi[1])
            
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.legend()


if __name__ == '__main__':
    
    folderDataset = rootDataPath + "/CFM2/datasets/"
    folderTrain = rootDataPath + "/CFM2/training/"
    
    Nrb = 140
    archId = 'huge'
    load_flag = 'S'
    suffix = "all"
    suffix_incomplete_case = "6x6_nobottom"
    nX = 36
    
    
    Xmask_list = {'all' : np.arange(nX), 
                  '35' : np.arange(nX - 1),
                  '4x4' : np.array([7,8,9,10,13,14,15,16,19,20,21,22,25,26,27,28]),
                  '2x2' : np.array([14,15,20,21]), 
                  '6x6_nobottom' : np.array([6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,
                                             25,26,27,28,29,30,31,32,33,34,35]),
                  '6x6_2nobottom' : np.array([12,13,14,15,16,17,18,19,20,21,22,23,24,
                                             25,26,27,28,29,30,31,32,33,34,35]),
                  '6x6_2nobottomright' : np.array([12,13,14,15,16,18,19,20,21,22,24,
                                             25,26,27,28,29,30,31,32,33,34]),
                  '6x6_2nobottom2right' : np.array([12,13,14,15,18,19,20,21,24,
                                             25,26,27,29,30,31,32,33]),
                  '6x6_nobottomright' : np.array([6,7,8,9,10,12,13,14,15,16,18,19,20,21,22,24,
                                             25,26,27,28,30,31,32,33,34]),
                  '4x4_nobottom' : np.array([13,14,15,16,19,20,21,22,25,26,27,28]),
                  '4x4_nobottom' : np.array([13,14,15,16,19,20,21,22,25,26,27,28])}
    
      
    nameXY_train = folderDataset +  'XY_train.hd5'
    nameXY_test = folderDataset +  'XY_test.hd5'
    
    net = standardNets[archId]
    net.nY = Nrb
    net.nX = nX
    net.archId =  archId
    net.files['weights'] = folderTrain + 'model_weights_%s_%s_%d_%s.hdf5'%(archId,load_flag,Nrb,suffix)
    net.files['net_settings'] =  folderTrain + 'model_net_%s_%s_%d_%s.txt'%(archId,load_flag,Nrb,suffix)
    net.files['prediction'] = folderTrain + 'model_prediction_%s_%s_%d_%s.txt'%(archId,load_flag,Nrb,suffix)
    net.files['scaler'] = folderTrain + 'scalers_%s_%s.txt'%(load_flag, suffix)
    net.files['XY_test'] = nameXY_test
    net.files['XY_train'] = nameXY_train
    net.files['Wbasis'] = folderDataset + "Wbasis.hd5"
    
    error, error_stats, error_DNN = compute_DNN_error(net, 'Y_%s'%load_flag, Xmask_list[suffix_incomplete_case])
    
    error_POD = compute_POD_error(net, load_flag)
    
    error_total = error_POD + error_DNN
    print(error_POD, error_DNN, error_total)
    
    plot_curve_training(net, 'loss', [(error_DNN, "error_DNN"), (error_total, "error_total")])