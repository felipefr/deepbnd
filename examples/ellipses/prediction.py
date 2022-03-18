#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 18:40:49 2022

@author: felipe
"""

import sys, os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import dolfin as df
# import matplotlib.pyplot as plt
# from ufl import nabla_div
import numpy as np

from deepBND.__init__ import *
# import deepBND.creation_model.training.wrapper_tensorflow as mytf
from deepBND.creation_model.training.net_arch import NetArch
# import deepBND.core.data_manipulation.utils as dman
import deepBND.core.data_manipulation.wrapper_h5py as myhd
from deepBND.core.fenics_tools.enriched_mesh import EnrichedMesh 
from deepBND.creation_model.prediction.NN_elast import NNElast
from deepBND.examples.ellipses.build_rb import mapSinusCosinus


class NNElast_angleMapSinusCosinus(NNElast):
    def getPermY(self, n = 6, m = 2):  
        ## it may be changed (permY)
        # the permY below is only valid for the ordenated radius (inside to outsid)
        # permY = np.array([2,0,3,1,12,10,8,4,13,5,14,6,15,11,9,7,30,28,26,24,22,16,31,17,32,18,33,19,34,20,35,29,27,25,23,21])
        # the permY below is only valid for the radius ordenated by rows and columns (below to top {left to right})
    
        perm = np.array([[(n-1-j)*n + i for j in range(n)] for i in range(n)]).flatten()  # note that (i,j) -> (Nx-j-1,i)
        perm_star = np.zeros(len(perm)*m).astype('int')
        
        for k in range(m):
            perm_star[k::m] = m*perm + k
            
        return perm_star 
        
def predictBCs(namefiles, net):
    
    labels = net.keys()
    
    nameMeshRefBnd, nameWbasis, paramRVEname, bcs_namefile = namefiles
  
    # loading boundary reference mesh
    Mref = EnrichedMesh(nameMeshRefBnd)
    Vref = df.VectorFunctionSpace(Mref,"CG", 1)
    
    # loading the DNN model
    paramRVEdata = myhd.loadhd5(paramRVEname, 'param')
    
    model = NNElast_angleMapSinusCosinus(nameWbasis, net, net['A'].nY)
    
    S_p = model.predict(mapSinusCosinus(paramRVEdata[:,:,4]), Vref)
    
    myhd.savehd5(bcs_namefile, S_p, ['u0','u1','u2'], mode = 'w')

standardNets = {'big': NetArch([300, 300, 300], 3*['swish'] + ['sigmoid'], 5.0e-4, 0.9, [0.0] + 3*[0.0] + [0.0], 0.0),
         'small':NetArch([40, 40, 40], 3*['swish'] + ['linear'], 5.0e-4, 0.8, [0.0] + 3*[0.001] + [0.0], 1.0e-8),
         'big_classical': NetArch([300, 300, 300], 3*['swish'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.005] + [0.0], 1.0e-8) }

if __name__ == '__main__':
  
    labels = ['A', 'S']
  
    archId = 'big_classical'
    Nrb = 140
    nX = 72
    
    suffix = '_test'
    
    folder = rootDataPath + "/ellipses/"
    folderTrain = folder + 'training_cluster/'
    folderBasis = folder + 'dataset_cluster/'
    folderPrediction = folder + "prediction_cluster/"
    nameMeshRefBnd = folderBasis + 'boundaryMesh.xdmf'
    nameWbasis = folderBasis +  'Wbasis.hd5'
    paramRVEname = folderPrediction + 'paramRVEdataset{0}.hd5'.format(suffix)
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

