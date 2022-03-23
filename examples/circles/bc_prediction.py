#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 00:54:51 2022

@author: felipe
"""

import sys, os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import dolfin as df
import matplotlib.pyplot as plt
from ufl import nabla_div
import numpy as np

from deepBND.__init__ import *
from deepBND.creation_model.training.net_arch import standardNets
import deepBND.core.data_manipulation.utils as dman
import deepBND.core.data_manipulation.wrapper_h5py as myhd
from deepBND.core.fenics_tools.enriched_mesh import EnrichedMesh 
from deepBND.creation_model.prediction.NN_elast import NNElast
    
def predictBCs(namefiles, net):
    
    labels = net.keys()
    
    nameMeshRefBnd, nameWbasis, paramRVEname, bcs_namefile = namefiles
  
    # loading boundary reference mesh
    Mref = EnrichedMesh(nameMeshRefBnd)
    Vref = df.VectorFunctionSpace(Mref,"CG", 1)
    # normal = FacetNormal(Mref)
    # volMref = 4.0
    
    # loading the DNN model
    paramRVEdata = myhd.loadhd5(paramRVEname, 'param')
    
    model = NNElast(nameWbasis, net, net['A'].nY)
    S_p = model.predict(paramRVEdata[:,:,2], Vref)
    
    myhd.savehd5(bcs_namefile, S_p, ['u0','u1','u2'], mode = 'w')

if __name__ == '__main__':
  
    labels = ['A', 'S']
  
    archId = 'big'
    Nrb = 140
    nX = 36
    
    folder = rootDataPath + "/paper/"
    folderTrain = folder + 'training_cluster/'
    folderBasis = folder + 'dataset_cluster/'
    folderPrediction = folder + "prediction_cluster/"
    nameMeshRefBnd = folderBasis + 'boundaryMesh.xdmf'
    nameWbasis = folderBasis +  'Wbasis.hd5'
    paramRVEname = folderPrediction + 'paramRVEdataset.hd5'
    bcs_namefile = folderPrediction + 'bcs_{0}_{1}.hd5'.format(archId, Nrb)
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


                                                                               