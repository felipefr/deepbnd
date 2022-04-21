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
from mpi4py import MPI

from deepBND.__init__ import *
import deepBND.core.data_manipulation.wrapper_h5py as myhd
from deepBND.core.fenics_tools.enriched_mesh import EnrichedMesh 
from deepBND.core.multiscale.micro_model_dnn import MicroConstitutiveModelDNN
from deepBND.core.multiscale.mesh_RVE import buildRVEmesh

# split BC prediction and paramRVEname

def predictTangents(modelBnd, namefiles, createMesh, meshSize):
    
    nameMeshRefBnd, paramRVEname, tangentName, BCname, meshMicroName = namefiles
    
    # loading boundary reference mesh
    Mref = EnrichedMesh(nameMeshRefBnd)
    Vref = df.VectorFunctionSpace(Mref,"CG", 1)
    
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
        
        if(os.path.exists(meshMicroName_i)):
            if(createMesh):
                print("mesh exists : jumping the calculation")
                continue
            else:
                pass
        else:
            buildRVEmesh(paramRVEdata[i,:,:], meshMicroName_i, isOrdered = False, size = meshSize)
    
        end = timer()
        print("time expended in meshing ", end - start)
    
        microModel = MicroConstitutiveModelDNN(meshMicroName_i, param, modelBnd) 
        
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
        Itangent[i,:,:] = microModel.getTangent()
        
        if(i%10 == 0):
            f.flush()    
            sys.stdout.flush()
            
    f.close()

# for i in {0..31}; do nohup python tangents_predictions_simplified.py $i > log_$i.txt & done
if __name__ == '__main__':
    
    run = int(sys.argv[1])
    
    suffixTangent = ''
    modelBnd = 'dnn'
    meshSize = 'reduced'
    createMesh = True
    suffixBC = ''
    suffix = ""

    if(modelBnd == 'dnn'):
        modelDNN = '' # underscore included before
    else:
        modelDNN = ''
               


    folder = rootDataPath + "/ellipses/"
    folderPrediction = folder + 'prediction_fresh_test/'
    folderMesh = folderPrediction + 'meshes/'
    paramRVEname = folderPrediction + 'paramRVEdataset{0}.hd5'.format(suffixBC) 
    nameMeshRefBnd = folderPrediction + 'boundaryMesh.xdmf'
    tangentName = folderPrediction + 'tangents_{0}.hd5'.format(modelBnd + modelDNN + suffixTangent)
    BCname = folderPrediction + 'bcs{0}{1}.hd5'.format(modelDNN,suffixBC) 
    meshMicroName = folderMesh + 'mesh_micro_{0}_{1}.xdmf'

    
    
    if(run == -1): # preparation paramRVE (splitting)
        numruns = int(sys.argv[2])
        # numruns = 10
        
        ns = len(myhd.loadhd5(paramRVEname, 'id'))
        
        size = int(np.floor(ns/numruns)) 
    
        labels = ['id', 'param']
        
        indexesOutput = [ np.arange(i*size, (i+1)*size) for i in range(numruns)]
        indexesOutput[-1] = np.arange((numruns-1)*size, ns) # in case the division is not exact
        
        myhd.split(paramRVEname, indexesOutput, labels)

        if(modelBnd == 'dnn'):
            labels = ['u0', 'u1', 'u2']
            myhd.split(BCname, indexesOutput, labels)

        
    else:
        paramRVEname = paramRVEname[:-4] + '_split/part_%d.hd5'%run
        BCname = BCname[:-4] + '_split/part_%d.hd5'%run
        
        os.system("mkdir " + tangentName[:-4] + '_split')
        tangentName = tangentName[:-4] + '_split/part_%d.hd5'%run
        
        namefiles = [nameMeshRefBnd, paramRVEname, tangentName, BCname, meshMicroName]
        predictTangents(modelBnd, namefiles, createMesh, meshSize)
