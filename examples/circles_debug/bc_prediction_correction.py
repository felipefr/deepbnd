#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 00:54:51 2022

@author: felipe
"""

import sys, os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt
import numpy as np

from deepBND.__init__ import *
import deepBND.core.data_manipulation.utils as dman
import deepBND.core.data_manipulation.wrapper_h5py as myhd
from deepBND.core.fenics_tools.enriched_mesh import EnrichedMesh
from deepBND.creation_model.training.net_arch import NetArch
from deepBND.creation_model.prediction.NN_elast import NNElast
import deepBND.core.elasticity.fenics_utils as feut

import dolfin as df

standardNets = {'big_A': NetArch([300, 300, 300], 3*['swish'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.005] + [0.0], 1.0e-8),
                'small_A':NetArch([40, 40, 40], 3*['swish'] + ['linear'], 5.0e-4, 0.8, [0.0] + 3*[0.001] + [0.0], 1.0e-8), 
                'big_S': NetArch([300, 300, 300], 3*['swish'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.005] + [0.0], 1.0e-8),
                'small_S':NetArch([40, 40, 40], 3*['swish'] + ['linear'], 5.0e-4, 0.8, [0.0] + 3*[0.001] + [0.0], 1.0e-8)}        

def predictBCs(namefiles, net, max_ns = 50):
    
    labels = net.keys()
    
    nameMeshRefBnd, nameWbasis, paramRVEname, bcs_namefile = namefiles
  
    # loading boundary reference mesh
    Mref = EnrichedMesh(nameMeshRefBnd)
    Vref = df.VectorFunctionSpace(Mref,"CG", 1)
    # normal = FacetNormal(Mref)
    # volMref = 4.0
    
    # loading the DNN model
    paramRVEdata = myhd.loadhd5(paramRVEname, 'param')[:max_ns]
    
    model = NNElast(nameWbasis, net, net['A'].nY)
    S_p = model.predict(paramRVEdata[:,:,2], Vref)
    
    myhd.savehd5(bcs_namefile, S_p, ['u0','u1','u2'], mode = 'w')

if __name__ == '__main__':
  
    labels = ['A', 'S']
  
    archId = 'big'
    Nrb = 140
    nX = 36
    
    folder = rootDataPath + "/DEBUG/"
    folderTrain = folder + 'training/'
    folderBasis = folder + 'dataset_recreation/'
    folderPrediction = folder + "prediction/"
    nameMeshRefBnd = folderBasis + 'boundaryMesh.xdmf'
    nameWbasis = folderBasis +  'Wbasis.hd5'
    paramRVEname = folderPrediction + 'paramRVEdataset.hd5'
    bcs_namefile = folderPrediction + 'bcs_poor_correc.hd5'
    nameScaleXY = {}
    
    net = {}
    
    for l in labels:
        net[l] = standardNets[archId + '_' + l] 
        net[l].nY = Nrb
        net[l].nX = nX
        net[l].files['weights'] = folderTrain + 'models_weights_%s_%s_%d.hdf5'%(archId, l,Nrb)
        net[l].files['scaler'] = folderTrain + 'scaler_%s_%d.txt'%(l, Nrb)

    namefiles = [nameMeshRefBnd, nameWbasis, paramRVEname, bcs_namefile]
    
    predictBCs(namefiles, net)


                                                                               