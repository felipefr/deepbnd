#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 18:40:49 2022

@author: felipe
"""
"""
This file is part of deepBND, a data-driven enhanced boundary condition implementaion for 
computational homogenization problems, using RB-ROM and Neural Networks.
Copyright (c) 2020-2023, Felipe Rocha.
See file LICENSE.txt for license information. 
Please cite this work according to README.md.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or <felipe.f.rocha@gmail.com>
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
from deepBND.creation_model.prediction.NN_elast import NNElast

import dolfin as df # apparently this import should be after the tensorflow stuff
        
def predictBCs(namefiles, net):
    
    labels = net.keys()
    
    nameMeshRefBnd, nameWbasis, paramRVEname, bcs_namefile = namefiles
  
    # loading boundary reference mesh
    Mref = EnrichedMesh(nameMeshRefBnd)
    Vref = df.VectorFunctionSpace(Mref,"CG", 1)
    
    # loading the DNN model
    paramRVEdata = myhd.loadhd5(paramRVEname, 'param')
    
    model = NNElast(nameWbasis, net, net['A'].nY)
    
    S_p = model.predict(paramRVEdata[:,:,2], Vref)
    
    myhd.savehd5(bcs_namefile, S_p, ['u0','u1','u2'], mode = 'w')

standardNets = {'small':NetArch([40, 40, 40], 3*['swish'] + ['linear'], 5.0e-4, 0.8, [0.0] + 3*[0.001] + [0.0], 1.0e-8),
               'big': NetArch([300, 300, 300], 3*['swish'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.005] + [0.0], 1.0e-8) }

if __name__ == '__main__':
  
    labels = ['A', 'S']
  
    archId = 'big'
    Nrb = 140
    nX = 36
    
    suffixParam = ''
    suffix = ''
    
    folder = rootDataPath + "/paper/"
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
        net[l].files['weights'] = folderTrain + 'model_weights_%s.hdf5'%(l)
        net[l].files['scaler'] = folderTrain + 'scaler_%s.txt'%(l)

    namefiles = [nameMeshRefBnd, nameWbasis, paramRVEname, bcs_namefile]
    
    predictBCs(namefiles, net)

