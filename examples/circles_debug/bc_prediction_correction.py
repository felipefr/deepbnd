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
import deepBND.core.elasticity.fenics_utils as feut

import dolfin as df

# standardNets = {'big_A': NetArch([300, 300, 300], 3*['swish'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.005] + [0.0], 1.0e-8),
#                 'small_A':NetArch([40, 40, 40], 3*['swish'] + ['linear'], 5.0e-4, 0.8, [0.0] + 3*[0.001] + [0.0], 1.0e-8), 
#                 'big_S': NetArch([300, 300, 300], 3*['swish'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.005] + [0.0], 1.0e-8),
#                 'small_S':NetArch([40, 40, 40], 3*['swish'] + ['linear'], 5.0e-4, 0.8, [0.0] + 3*[0.001] + [0.0], 1.0e-8)}
    

standardNets = {'big': NetArch([300, 300, 300], 3*['swish'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.005] + [0.0], 1.0e-8),
                'small':NetArch([40, 40, 40], 3*['swish'] + ['linear'], 5.0e-4, 0.8, [0.0] + 3*[0.001] + [0.0], 1.0e-8)}

class NNElast:
    
    def __init__(self, nameWbasis, net, Nrb):
        
        self.Nrb = Nrb
        self.labels = net.keys()
        
        self.Wbasis = {}
        self.scalerX = {}; self.scalerY = {}
        self.model = {}

        for l in self.labels:
            self.Wbasis[l] = myhd.loadhd5(nameWbasis, 'Wbasis_%s'%l)
            self.scalerX[l], self.scalerY[l]  = dman.importScale(net[l].files['scaler'], net[l].nX, net[l].nY)
            self.model[l] = net[l].getModel()
            self.model[l].load_weights(net[l].files['weights'])

    def predict(self, X, Vref):
        X_s = {}  
        Y_p = {}; S_p = {}
        
        for l in self.labels:
            X_s[l] = self.scalerX[l].transform(X)
            Y_p[l] = self.scalerY[l].inverse_transform(self.model[l].predict(X_s[l]))
            S_p[l] = Y_p[l] @ self.Wbasis[l][:self.Nrb,:]

        X_axialY_s = self.scalerX['A'].transform(X[:,self.getPermY()]) ### permY performs a counterclockwise rotation
        Y_p_axialY = self.scalerY['A'].inverse_transform(self.model['A'].predict(X_axialY_s)) 
        
        
        theta = 3*np.pi/2.0 # 'minus HalfPi'
        piola_mat = feut.PiolaTransform_rotation_matricial(theta, Vref)
        S_p_axialY = Y_p_axialY @ self.Wbasis['A'][:self.Nrb,:] @ piola_mat.T #changed
        
        return [S_p['A'], S_p_axialY, S_p['S']]
        
    def getPermY(self):  
        ## it may be changed (permY)
        # the permY below is only valid for the ordenated radius (inside to outsid)
        # permY = np.array([2,0,3,1,12,10,8,4,13,5,14,6,15,11,9,7,30,28,26,24,22,16,31,17,32,18,33,19,34,20,35,29,27,25,23,21])
        # the permY below is only valid for the radius ordenated by rows and columns (below to top {left to right})
        return np.array([[(5-j)*6 + i for j in range(6)] for i in range(6)]).flatten() # note that (i,j) -> (Nx-j-1,i)
        

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
        # net[l] = standardNets[archId + '_' + l] 
        net[l] = standardNets[archId] 
        net[l].nY = Nrb
        net[l].nX = nX
        net[l].files['weights'] = folderTrain + 'models_weights_%s_%s_%d.hdf5'%(archId, l,Nrb)
        net[l].files['scaler'] = folderTrain + 'scaler_%s_%d.txt'%(l, Nrb)

    namefiles = [nameMeshRefBnd, nameWbasis, paramRVEname, bcs_namefile]
    
    predictBCs(namefiles, net)


                                                                               