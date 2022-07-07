#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 14:25:42 2022

@author: felipe
"""

import sys, os
import numpy as np
import dolfin as df 
import matplotlib.pyplot as plt
from ufl import nabla_div
from timeit import default_timer as timer

from deepBND.__init__ import *
import fetricks.data_manipulation.wrapper_h5py as myhd
from fetricks.fenics.mesh.mesh import Mesh 
from deepBND.core.multiscale.micro_model_gen import MicroConstitutiveModelGen
# from deepBND.core.multiscale.micro_model_dnn import MicroConstitutiveModelDNN
from deepBND.core.multiscale.mesh_RVE import buildRVEmesh

# split BC prediction and paramRVEname

def predictTangents(modelBnd, namefiles, createMesh, meshSize):
    
    nameMeshRefBnd, paramRVEname, tangentName, BCname, meshMicroName = namefiles
    
    # loading boundary reference mesh
    Mref = Mesh(nameMeshRefBnd)
    Vref = df.VectorFunctionSpace(Mref,"CG", 2)
    
    dxRef = df.Measure('dx', Mref) 
    
    # defining the micro model
    paramRVEdata = myhd.loadhd5(paramRVEname, 'param')
    ns = len(paramRVEdata) 
    
    # Id here have nothing to with the position of the RVE in the body. Solve it later
    if(myhd.checkExistenceDataset(paramRVEname, 'id')):    
        ids = myhd.loadhd5(paramRVEname, 'id')
    else:
        ids = -1*np.ones(ns).astype('int')
        
    if(myhd.checkExistenceDataset(paramRVEname, 'center')):    
        centers = myhd.loadhd5(paramRVEname, 'center')[ids,:]
    else:
        centers = np.zeros((ns,2))
       
    os.system('rm ' + tangentName)
    Iid_tangent_center, f = myhd.zeros_openFile(tangentName, [(ns,), (ns,3,3), (ns,2)],
                                           ['id', 'tangent','center'], mode = 'w')
    
    Iid, Itangent, Icenter = Iid_tangent_center
    
    if(modelBnd == 'dnn'):
        u0_p = myhd.loadhd5(BCname, 'u0')
        u1_p = myhd.loadhd5(BCname, 'u1')
        u2_p = myhd.loadhd5(BCname, 'u2')
        
    for i in range(ns):
        Iid[i] = ids[i]
        
        contrast = 10.0
        E2 = 1.0
        nu = 0.3
        param = [nu,E2*contrast,nu,E2]
        print(paramRVEname, i, ids[i])
        meshMicroName_i = meshMicroName.format(int(Iid[i]), meshSize)
    
        start = timer()
        
        buildRVEmesh(paramRVEdata[i,:,:], meshMicroName_i, isOrdered = False, size = meshSize, 
                     NxL = 4, NyL = 4, maxOffset = 2, lcar = 3/30)
        
        end = timer()
        print("time expended in meshing ", end - start)
    
        # microModel = MicroConstitutiveModelDNN(meshMicroName_i, param, modelBnd) 
        microModel = MicroConstitutiveModelGen(meshMicroName_i, param, modelBnd)
        
        if(modelBnd == 'dnn'):
            microModel.others['uD'] = df.Function(Vref) 
            microModel.others['uD0_'] = u0_p[i] # it was already picked correctly
            microModel.others['uD1_'] = u1_p[i] 
            microModel.others['uD2_'] = u2_p[i]
        elif(modelBnd == 'lin'):
            microModel.others['uD'] = df.Function(Vref) 
            microModel.others['uD0_'] = np.zeros(Vref.dim())
            microModel.others['uD1_'] = np.zeros(Vref.dim())
            microModel.others['uD2_'] = np.zeros(Vref.dim())
            
        
        Icenter[i,:] = centers[i,:]
        # Itangent[i,:,:] = microModel.getTangent()
        Itangent[i,:,:] = microModel.getHomogenisation()['tangentL']
        
        if(i%10 == 0):
            f.flush()    
            sys.stdout.flush()
            
    f.close()

# for i in {0..31}; do nohup python tangents_predictions_simplified.py $i > log_$i.txt & done
if __name__ == '__main__':
    
    run = 0
    
    suffixTangent = 'full'
    modelBnd = 'per'
    meshSize = 'full'
    createMesh = True
    suffixBC = ''
    suffix = ""

    if(modelBnd == 'dnn'):
        modelDNN = '' # underscore included before
    else:
        modelDNN = ''
               

    folder = rootDataPath + "/review2/"
    folderPrediction = folder + 'prediction/'
    folderMesh = folderPrediction + 'meshes/'
    paramRVEname = folderPrediction + 'paramRVEdataset.hd5' 
    nameMeshRefBnd = folderPrediction + 'boundaryMesh.xdmf'
    tangentName = folderPrediction + 'tangents_{0}.hd5'.format(modelBnd + modelDNN + suffixTangent)
    BCname = folderPrediction + 'bcs{0}{1}.hd5'.format(modelDNN,suffixBC) 
    meshMicroName = folderMesh + 'mesh_micro_{0}_{1}.xdmf'

    namefiles = [nameMeshRefBnd, paramRVEname, tangentName, BCname, meshMicroName]
    predictTangents(modelBnd, namefiles, createMesh, meshSize)