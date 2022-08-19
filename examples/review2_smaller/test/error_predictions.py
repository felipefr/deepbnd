#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 11:37:12 2022

@author: felipefr
"""

import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np


from deepBND.__init__ import * 
import deepBND.creation_model.training.wrapper_tensorflow as mytf
import deepBND.creation_model.RB.RB_utils as rbut
import tensorflow as tf
# from deepBND.creation_model.training.net_arch import standardNets
from deepBND.creation_model.training.net_arch import NetArch
import deepBND.core.data_manipulation.utils as dman
import deepBND.core.data_manipulation.wrapper_h5py as myhd
from fetricks.fenics.mesh.mesh import Mesh 

import dolfin as df # apparently this import should be after the tensorflow stuff

from deepBND.creation_model.prediction.NN_elast_positions_6x6 import NNElast_positions_6x6
import fetricks as ft

standardNets = {'big_A': NetArch([300, 300, 300], 3*['swish'] + ['linear'], 5.0e-4, 0.1, 5*[0.0], 1.0e-8),
                'big_S': NetArch([300, 300, 300], 3*['swish'] + ['linear'], 5.0e-4, 0.1, 5*[0.0], 1.0e-8),            
                'big_tri_A': NetArch([150, 300, 500], 3*['swish'] + ['linear'], 5.0e-4, 0.1, 2*[0.0] + [0.005] + 2*[0.0], 1.0e-8),
                'big_tri_S': NetArch([150, 300, 500], 3*['swish'] + ['linear'], 5.0e-4, 0.1, 2*[0.0] + [0.005] + 2*[0.0], 1.0e-8)}


def getScaledXY(net, Ylabel):
    scalerX, scalerY = dman.importScale(net.files['scaler'], nX, Nrb, scalerType = 'MinMax11')
    Xbar, Ybar = dman.getDatasetsXY(nX, Nrb, net.files['XY'], scalerX, scalerY, Ylabel = Ylabel)[0:2]
    
    return Xbar, Ybar, scalerX, scalerY
    
def loadYlist(net, Ylabel):
    dummy1, Ybar, dummy2, scalerY = getScaledXY(net, Ylabel)
    Y = scalerY.inverse_transform(Ybar)
    return Y 

def predictYlist(net, Ylabel):
    model = net.getModel()   
    model.load_weights(net.files['weights'])
    
    Xbar, dummy1, dummy2, scalerY = getScaledXY(net, Ylabel)
    Yp = scalerY.inverse_transform(model.predict(Xbar)) 

    return Yp

def compute_DNN_error(net, Ylabel):

    Y = loadYlist(net, Ylabel)
    Yp = predictYlist(net, Ylabel)
            
    error = np.sum((Yp - Y)**2, axis = 1) # all modes along each snapshot
    
    error_dict = {'snapshots': error, 
                  "mean": np.mean(error),
                  "std": np.std(error),
                  "max": np.max(error),
                  "min": np.min(error)}
    
    return error_dict['mean'], error_dict


def compute_POD_error(net, loadType, NS):
    eig = myhd.loadhd5(net.files['Wbasis'], 'sig_%s'%loadType)   
        
    errorPOD = np.sum(eig[net.nY:])/NS
    
    return errorPOD

def compute_total_error_bruteForce(net, loadtype, suffix, bndMeshname, snapshotsname, paramRVEname):
    
    Ylabel = "Y_%s"%loadtype
    
    Yp = predictYlist(net, Ylabel)
    # Yp = loadYlist(net, Ylabel)
    
    Mref = Mesh(bndMeshname)
    Vref = df.VectorFunctionSpace(Mref,"CG", 2)
    
    dsRef = df.Measure('ds', Mref) 
    
    Wbasis = myhd.loadhd5( net.files['Wbasis'], 'Wbasis_%s'%loadtype)
    ids = myhd.loadhd5(paramRVEname, "ids").astype('int')
    Isol = myhd.loadhd5( snapshotsname, 'solutions_%s_%s'%(suffix, loadtype))[ids, :] 
    
    error = rbut.getMSE([net.nY], Yp, Wbasis, Isol, Vref, dsRef, lambda u, v, dx: df.assemble(df.inner(u,v)*dx))
    
    return error


def compute_total_error_bruteForce_fast(net, loadtype, suffix, snapshotsname, paramRVEname):
    
    Ylabel = "Y_%s"%loadtype
    
    Yp = predictYlist(net, Ylabel)
    # Yp = loadYlist(net, Ylabel)
    
    Wbasis_M = myhd.loadhd5( net.files['Wbasis'], ['Wbasis_%s'%loadtype, 'massMat'])
    ids = myhd.loadhd5(paramRVEname, "ids").astype('int')
    Isol = myhd.loadhd5( snapshotsname, 'solutions_%s_%s'%(suffix, loadtype))[ids, :] 
    
    error = rbut.getMSE_fast([net.nY], Yp, Wbasis_M, Isol)
    
    return error


def compute_DNN_error_bruteForce_fast(net, loadtype, suffix, snapshotsname, paramRVEname):
    
    Ylabel = "Y_%s"%loadtype
    
    Yp = predictYlist(net, Ylabel)
    Yt = loadYlist(net, Ylabel)
    
    Wbasis_M = myhd.loadhd5( net.files['Wbasis'], ['Wbasis_%s'%loadtype, 'massMat'])
    ids = myhd.loadhd5(paramRVEname, "ids").astype('int')
    
    error = rbut.getMSE_DNN_fast([net.nY], Yp, Yt, Wbasis_M)
    
    return error





folder = rootDataPath + '/review2_smaller/'
folderDataset = folder + 'dataset/'
folderPrediction = folder + 'prediction/'
folderTrain = folder + 'training/'
folderBasis = folder + 'dataset/'

nameMeshRefBnd = folderBasis + 'boundaryMesh.xdmf'
snapshotsname = folderDataset + 'snapshots.hd5'
paramRVEname = folderDataset + 'paramRVEdataset.hd5'

archId = 'big'
Nrb = 600
nX = 72

suffix = 'translation'

label = 'S' 

net = standardNets[archId + '_' +  label] 
net.nY = Nrb
net.nX = nX
net.files['XY'] = folderDataset + 'XY_%s.hd5'%suffix 
net.files['Wbasis'] = folderDataset + 'Wbasis_%s.hd5'%suffix
net.files['weights'] = folderTrain + 'models_weights_%s_%s_%d_%s.hdf5'%(archId, label, Nrb, suffix)
net.files['scaler'] = folderTrain + 'scaler_%s_%s.txt'%(suffix, label)
net.files['hist'] = folderTrain + 'models_weights_%s_%s_%d_%s_plot_history_val.txt'%(archId, label, Nrb, suffix) 
    
error_DNN, errors = compute_DNN_error(net, 'Y_%s'%label)
error_POD = compute_POD_error(net, label, 60000)
error_tot = error_POD + error_DNN
print(error_DNN)
print(error_POD)
print(error_tot)

print(0.25*np.min(np.loadtxt(net.files['hist']))) # indeed the weights were 4x than they should



error_tot_brute = compute_total_error_bruteForce_fast(net, label, suffix, snapshotsname, paramRVEname)
# error_tot_brute = compute_total_error_bruteForce_fast(net, label, suffix, nameMeshRefBnd, snapshotsname, paramRVEname)
print(error_tot_brute)

# paramRVEname_test = folderPrediction +  'paramRVEdataset_test.hd5'
# snapshotsname = folderDataset +  'snapshots.hd5'
# BCname = folderPrediction + 'bcs_fluctuations_big_test.hd5'
# bcs_namefile = folderPrediction + 'bcs_%s_bigtri_200_test.hd5'%suffix


print( compute_DNN_error_bruteForce_fast(net, label, suffix, snapshotsname, paramRVEname) )


