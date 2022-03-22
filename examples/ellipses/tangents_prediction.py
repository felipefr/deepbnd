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

# for i in {0..19}; do nohup python computeTangents_serial.py 24 $i 20 > log_ny24_full_per_run$i.py & done

comm = MPI.COMM_WORLD
comm_self = MPI.COMM_SELF

def predictTangents(num, num_runs, modelBnd, namefiles, createMesh, meshSize):
    
    nameMeshRefBnd, paramRVEname, tangentName, BCname, meshMicroName = namefiles
    
    # loading boundary reference mesh
    Mref = EnrichedMesh(nameMeshRefBnd,comm_self)
    Vref = df.VectorFunctionSpace(Mref,"CG", 1)
    
    dxRef = df.Measure('dx', Mref) 
    
    # defining the micro model
    
    paramRVEdata = myhd.loadhd5(paramRVEname, 'param')[run::num_runs]
    ns = len(paramRVEdata) # per rank
    
    # Id here have nothing to with the position of the RVE in the body. Solve it later
    if(myhd.checkExistenceDataset(paramRVEname, 'id')):    
        ids = myhd.loadhd5(paramRVEname, 'id')[run::num_runs].astype('int')
    else:
        ids = np.zeros(ns).astype('int')
        
    if(myhd.checkExistenceDataset(paramRVEname, 'center')):    
        centers = myhd.loadhd5(paramRVEname, 'center')[ids,:]
    else:
        centers = np.zeros((ns,2))
       
    os.system('rm ' + tangentName)
    Iid_tangent_center, f = myhd.zeros_openFile(tangentName, [(ns,), (ns,3,3), (ns,2)],
                                           ['id', 'tangent','center'], mode = 'w')
    
    Iid, Itangent, Icenter = Iid_tangent_center
    
    if(modelBnd == 'dnn'):
        u0_p = myhd.loadhd5(BCname, 'u0')[run::num_runs,:]
        u1_p = myhd.loadhd5(BCname, 'u1')[run::num_runs,:]
        u2_p = myhd.loadhd5(BCname, 'u2')[run::num_runs,:]
    
    for i in range(ns):
        Iid[i] = ids[i]
        
        contrast = 10.0
        E2 = 1.0
        nu = 0.3
        param = [nu,E2*contrast,nu,E2]
        print(run, i, ids[i])
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

if __name__ == '__main__':
    
    if(len(sys.argv)>1):
        run = int(sys.argv[1])
        num_runs = int(sys.argv[2])
    else:
        run = comm.Get_rank()
        num_runs = comm.Get_size()
    
    print('run, num_runs ', run, num_runs)
    
    suffixTangent = 'per_full'
    modelBnd = 'per'
    meshSize = 'full'
    createMesh = False
    suffixBC = '_test'
    suffix = "_full_test"

    # for i in {0..31}; do nohup python tangents_prediction.py $i 32 > log_full_per_test_$i.txt & done
    # for i in {0..9}; do nohup python tangents_prediction.py $i 10 > log_val_$i.txt & done

    # nohup mpiexec -n 8 python tangents_prediction.py log_val_mpiexec.txt &

    if(modelBnd == 'dnn'):
        modelDNN = '_big_80' # underscore included before
    else:
        modelDNN = ''
               
    folder = rootDataPath + "/ellipses/"
    folderPrediction = folder + 'prediction_cluster/'
    folderMesh = folderPrediction + 'meshes/'
    folderDataset = folder + 'dataset_cluster/'
    paramRVEname = folderPrediction + 'paramRVEdataset{0}.hd5'.format(suffixBC) 
    nameMeshRefBnd = folderDataset + 'boundaryMesh.xdmf'
    tangentName = folderPrediction + 'tangents_{0}_{1}.hd5'.format(modelBnd + modelDNN + suffix,run)
    BCname = folderPrediction + 'bcs{0}{1}.hd5'.format(modelDNN,suffixBC) 
    meshMicroName = folderMesh + 'mesh_micro_{0}_{1}.xdmf'

    namefiles = [nameMeshRefBnd, paramRVEname, tangentName, BCname, meshMicroName]
    
    predictTangents(run, num_runs, modelBnd, namefiles, createMesh, meshSize)
    

