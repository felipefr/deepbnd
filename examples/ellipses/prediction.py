#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 18:40:49 2022

@author: felipe
"""


import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np

from deepBND.__init__ import * 
import deepBND.creation_model.training.wrapper_tensorflow as mytf
# from deepBND.creation_model.training.net_arch import standardNets
from deepBND.creation_model.training.net_arch import NetArch
import deepBND.core.data_manipulation.utils as dman
import deepBND.core.data_manipulation.wrapper_h5py as myhd

from deepBND.core.fenics_tools.enriched_mesh import EnrichedMesh 
from deepBND.creation_model.prediction.NN_elast_ellipses import NNElast_ellipses

import dolfin as df # apparently this import should be after the tensorflow stuff


standardNets = {'big_A': NetArch([300, 300, 300], 3*['swish'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.005] + [0.0], 1.0e-8),
                'small_A':NetArch([40, 40, 40], 3*['swish'] + ['linear'], 5.0e-4, 0.8, [0.0] + 3*[0.001] + [0.0], 1.0e-8), 
                'big_S': NetArch([300, 300, 300], 3*['swish'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.005] + [0.0], 1.0e-8),
                'small_S':NetArch([40, 40, 40], 3*['swish'] + ['linear'], 5.0e-4, 0.8, [0.0] + 3*[0.001] + [0.0], 1.0e-8), 
                'big_nonclassical_A': NetArch([300, 300, 300], 3*['swish'] + ['sigmoid'], 5.0e-4, 0.9, [0.0] + 3*[0.0] + [0.0], 0.0),
                'big_nonclassical_S': NetArch([300, 300, 300], 3*['swish'] + ['sigmoid'], 5.0e-4, 0.9, [0.0] + 3*[0.0] + [0.0], 0.0)}  
        
def predictBCs(namefiles, net, ns_max = None):
    
    labels = net.keys()
    
    nameMeshRefBnd, nameWbasis, paramRVEname, bcs_namefile = namefiles
  
    # loading boundary reference mesh
    Mref = EnrichedMesh(nameMeshRefBnd)
    Vref = df.VectorFunctionSpace(Mref,"CG", 1)
    
    # loading the DNN model
    if(type(ns_max) == type(None)):
        paramRVEdata = myhd.loadhd5(paramRVEname, 'param')
    else:
        paramRVEdata = myhd.loadhd5(paramRVEname, 'param')[:ns_max]
    
    model = NNElast_ellipses(nameWbasis, net, net['A'].nY)
    
    S_p = model.predict(paramRVEdata[:,:,4], Vref)
    
    myhd.savehd5(bcs_namefile, S_p, ['u0','u1','u2'], mode = 'w')


if __name__ == '__main__':
  
    labels = ['A', 'S']
  
    archId = 'big'
    Nrb = 140
    nX = 72
    
    suffixBC = ''
    
    
    folder = rootDataPath + "/ellipses/"
    folderTrain = folder + 'training_fresh_data_augmentation/'
    folderBasis = folder + 'training_fresh_data_augmentation/'
    folderPrediction = folder + "prediction_fresh_test/"
    nameMeshRefBnd = folderBasis + 'boundaryMesh.xdmf'
    nameWbasis = folderBasis +  'Wbasis.hd5'
    paramRVEname = folderPrediction + 'paramRVEdataset.hd5'
    bcs_namefile = folderPrediction + 'bcs{0}.hd5'.format(suffixBC)
    nameScaleXY = {}
    
    net = {}
    
    for l in labels:
        net[l] = standardNets[archId + '_' +  l] 
        net[l].nY = Nrb
        net[l].nX = nX
        net[l].files['weights'] = folderTrain + 'models/weights_%s_%s_%d.hdf5'%(archId, l, Nrb)
        net[l].files['scaler'] = folderTrain + 'scalers/scaler_%s_%d.txt'%(l, Nrb)

    namefiles = [nameMeshRefBnd, nameWbasis, paramRVEname, bcs_namefile]
    
    predictBCs(namefiles, net)

