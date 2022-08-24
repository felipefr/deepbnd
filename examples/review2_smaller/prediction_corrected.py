#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 15:11:26 2022

@author: felipefr
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

from fetricks.fenics.mesh.mesh import Mesh 
from deepBND.creation_model.prediction.NN_elast_positions_6x6 import NNElast_positions_6x6

import dolfin as df # apparently this import should be after the tensorflow stuff


standardNets = {'big_A': NetArch([300, 300, 300], 3*['swish'] + ['linear'], 5.0e-4, 0.1, 5*[0.0], 1.0e-8),
                'big_S': NetArch([300, 300, 300], 3*['swish'] + ['linear'], 5.0e-4, 0.1, 5*[0.0], 1.0e-8),            
                'big_tri_A': NetArch([150, 300, 500], 3*['swish'] + ['linear'], 5.0e-4, 0.1, 2*[0.0] + [0.005] + 2*[0.0], 1.0e-8),
                'big_tri_S': NetArch([150, 300, 500], 3*['swish'] + ['linear'], 5.0e-4, 0.1, 2*[0.0] + [0.005] + 2*[0.0], 1.0e-8)}

        
def predictBCs(namefiles, net, param_subset = None):
    
    labels = net.keys()
    
    nameMeshRefBnd, nameWbasis, paramRVEname, bcs_namefile, snapshotsname = namefiles
  
    # loading boundary reference mesh
    Mref = Mesh(nameMeshRefBnd)
    Vref = df.VectorFunctionSpace(Mref,"CG", 2)
    
    ids = myhd.loadhd5(paramRVEname, 'ids').flatten().astype('int')
    paramRVEdata = myhd.loadhd5(paramRVEname, 'param')

    model = NNElast_positions_6x6(nameWbasis, net, net['A'].nY, "MinMax11")
    
    Bten_ref = getBten_ref(snapshotsname, ids)
    # S_p = model.predict_correctedbyBten(paramRVEdata[:,:,0:2].reshape((len(paramRVEdata),-1)), Vref, Bten_ref)
    S_p = model.predict(paramRVEdata[:,:,0:2].reshape((len(paramRVEdata),-1)), Vref)
    
    
    myhd.savehd5(bcs_namefile, [ids] + S_p , ['ids', 'u0','u1','u2'], mode = 'w')

def getBten_ref(snapshotsname, ids):
    Bten_ref_A = myhd.loadhd5(snapshotsname, "B_A")[ids]
    Bten_ref_S = myhd.loadhd5(snapshotsname, "B_S")[ids]
     
    return {'A': Bten_ref_A, 'S': Bten_ref_S}

if __name__ == '__main__':
  
    labels = ['A', 'S']
  
    archId = 'big_tri'
    Nrb = 200
    nX = 72
    
    suffix = 'fluctuations'
    
    folder = rootDataPath + "/review2_smaller/"
    folderTrain = folder + 'training/'
    folderBasis = folder + 'dataset/'
    folderPrediction = folder + "prediction_test2/"
    nameMeshRefBnd = folderBasis + 'boundaryMesh.xdmf'
    nameSnapshots = folderBasis + 'snapshots.hd5'
    nameWbasis = folderBasis +  'Wbasis_%s_zerofied.hd5'%suffix
    paramRVEname = folderPrediction + 'paramRVEdataset_test.hd5'
    bcs_namefile = folderPrediction + 'bcs_%s_bigtri_200_test_deltaChanged.hd5'%suffix
    nameScaleXY = {}
    
    net = {}
    
    for l in labels:
        net[l] = standardNets[archId + '_' +  l] 
        net[l].nY = Nrb
        net[l].nX = nX
        net[l].files['weights'] = folderTrain + 'models_weights_%s_%s_%d_%s.hdf5'%(archId, l, Nrb, suffix)
        net[l].files['scaler'] = folderTrain + 'scaler_%s_%s.txt'%(suffix, l)

    namefiles = [nameMeshRefBnd, nameWbasis, paramRVEname, bcs_namefile, nameSnapshots]
    
    predictBCs(namefiles, net)

