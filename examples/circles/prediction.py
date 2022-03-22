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
        
def predictBCs(namefiles, net):
    
    labels = net.keys()
    
    nameMeshRefBnd, nameWbasis, paramRVEname, bcs_namefile = namefiles
  
    # loading boundary reference mesh
    Mref = EnrichedMesh(nameMeshRefBnd)
    Vref = df.VectorFunctionSpace(Mref,"CG", 1)
    
    # loading the DNN model
    paramRVEdata = myhd.loadhd5(paramRVEname, 'param')
    
    model = NNElast_ellipses(nameWbasis, net, net['A'].nY)
    
    S_p = model.predict(paramRVEdata[:,:,4], Vref)
    
    S_p = [4.0*s for s in S_p]
    myhd.savehd5(bcs_namefile, S_p, ['u0','u1','u2'], mode = 'w')

standardNets = {'big': NetArch([300, 300, 300], 3*['swish'] + ['sigmoid'], 5.0e-4, 0.9, [0.0] + 3*[0.0] + [0.0], 0.0),
         'small':NetArch([40, 40, 40], 3*['swish'] + ['linear'], 5.0e-4, 0.8, [0.0] + 3*[0.001] + [0.0], 1.0e-8),
         'big_classical': NetArch([300, 300, 300], 3*['swish'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.005] + [0.0], 1.0e-8) }

if __name__ == '__main__':
  
    labels = ['A', 'S']
  
    archId = 'big_classical'
    Nrb = 140
    nX = 72
    
    suffixParam = '_test'
    suffix = '_test_rotated_x4'
    
    
    folder = rootDataPath + "/ellipses/"
    folderTrain = folder + 'training_cluster/'
    folderBasis = folder + 'dataset_cluster/'
    folderPrediction = folder + "prediction_cluster/"
    nameMeshRefBnd = folderBasis + 'boundaryMesh.xdmf'
    nameWbasis = folderBasis +  'Wbasis.hd5'
    paramRVEname = folderPrediction + 'paramRVEdataset{0}.hd5'.format(suffixParam)
    bcs_namefile = folderPrediction + 'bcs_{0}_{1}{2}.hd5'.format(archId, Nrb, suffix)
    nameScaleXY = {}
    
    net = {}
    
    for l in labels:
        net[l] = standardNets[archId] 
        net[l].nY = Nrb
        net[l].nX = nX
        net[l].files['weights'] = folderTrain + 'models_weights_%s_%s_%d.hdf5'%(archId, l, Nrb)
        net[l].files['scaler'] = folderTrain + 'scaler_%s_%d.txt'%(l, Nrb)

    namefiles = [nameMeshRefBnd, nameWbasis, paramRVEname, bcs_namefile]
    
    predictBCs(namefiles, net)

