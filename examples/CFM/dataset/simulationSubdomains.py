#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 11:15:29 2022

@author: felipe
"""

import sys, os
import numpy as np
from numpy import isclose
import copy
from timeit import default_timer as timer
from mpi4py import MPI

from deepBND.__init__ import *

import deepBND.creation_model.training.wrapper_tensorflow as mytf
from deepBND.creation_model.training.net_arch import NetArch
from deepBND.creation_model.prediction.NN_elast import NNElast 

import dolfin as df
from deepBND.core.multiscale.mesh_RVE import buildRVEmesh
from deepBND.core.multiscale.micro_model_gen import MicroConstitutiveModelGen
import deepBND.core.elasticity.fenics_utils as fela
import deepBND.core.fenics_tools.wrapper_io as iofe
import deepBND.core.multiscale.misc as mtsm
from deepBND.core.fenics_tools.enriched_mesh import EnrichedMesh 
import deepBND.core.data_manipulation.wrapper_h5py as myhd
from deepBND.core.mesh.degenerated_rectangle_mesh import degeneratedBoundaryRectangleMesh
from deepBND.core.multiscale.mesh_RVE import paramRVE_default


comm = MPI.COMM_WORLD
comm_self = MPI.COMM_SELF

# comm = MPI.COMM_WORLD
# comm_self = MPI.COMM_SELF



standardNets = {'big': NetArch([300, 300, 300], 3*['swish'] + ['linear'], 5.0e-4, 0.1, [0.0] + 3*[0.005] + [0.0], 1.0e-8),
                'small':NetArch([40, 40, 40], 3*['swish'] + ['linear'], 5.0e-4, 0.8, [0.0] + 3*[0.001] + [0.0], 1.0e-8)}  

def predictBCs(features, folder):

    archId = 'big'
    Nrb = 140
    nX = 36
    
    folderDNN = folder + '/DNN/'
    nameMeshRefBnd = folderDNN + 'boundaryMesh.xdmf'
    nameWbasis = folderDNN +  'Wbasis.hd5'
    
    net = {}
    
    labels = ['A', 'S']
    for l in labels:
        net[l] = standardNets[archId] 
        net[l].nY = Nrb
        net[l].nX = nX
        net[l].files['weights'] = folderDNN + 'weights_%s.hdf5'%(l)
        net[l].files['scaler'] = folderDNN + 'scaler_%s.txt'%(l)
                  
    # loading boundary reference mesh
    Mref = EnrichedMesh(nameMeshRefBnd)
    Vref = df.VectorFunctionSpace(Mref,"CG", 1)
    
    
    model = NNElast(nameWbasis, net, Nrb)
    
    U = model.predict(features, Vref)
    
    return {'uD': df.Function(Vref), 'u0': U[0], 'u1': U[1], 'u2': U[2]}


def solve_snapshot(i, meshname, paramMaterial, opModel, snapshots, bcsDNN = None):
    start = timer()
    
    
    tangent, tangentL, sigma, sigmaL, eps, epsL = snapshots[2:]
    
    microModel = MicroConstitutiveModelGen(meshname, paramMaterial, opModel)
    
    if(opModel == 'dnn'):
        microModel.others['uD'] = bcsDNN['uD'] 
        microModel.others['uD0_'] = bcsDNN['u0'][i]
        microModel.others['uD1_'] = bcsDNN['u1'][i]  
        microModel.others['uD2_'] = bcsDNN['u2'][i] 
        
      
    Hom = microModel.computeHomogenisation([0,1])   
    tangent[i,:,:] = Hom['tangent']
    tangentL[i,:,:] = Hom['tangentL']
    sigma[i,:,:] = Hom['sigma']
    sigmaL[i,:,:] = Hom['sigmaL']
    eps[i,:,:] = Hom['eps']
    epsL[i,:,:] = Hom['epsL']

    end = timer()

    print("concluded in ", end - start)      
    

def buildSnapshots(paramMaterial, filesnames, opModel, createMesh, run, num_runs, i0):
    paramRVEname, snapshotsname, meshname, folder = filesnames
    
    ns = len(myhd.loadhd5(paramRVEname, 'param')[:,0])
    
    ids_run = np.arange(run,ns,num_runs).astype('int')
    nrun = len(ids_run)
    
    os.system('rm ' + snapshotsname)
    snapshots, fsnaps = myhd.zeros_openFile(filename = snapshotsname,  
                                            shape = 2*[(nrun,)] + 6*[(nrun,3,3)],
                                            label = ['id', 'id_local', 'tangent', 'tangentL', 'sigma', 'sigmaL', 'eps', 'epsL'], mode = 'w-')
    
    id_, id_local = snapshots[0:2]
             
    paramRVEdata, id_local_paramRVE, id_paramRVE = myhd.loadhd5(paramRVEname, ['param', 'id_local', 'id']) 
    
    bcsDNN = None

    if(opModel == 'dnn'): # first run
        bcsDNN = predictBCs(paramRVEdata[:,:,2], folder)  

    
    for i in range(i0,nrun):
        
        id_[i] = id_paramRVE[i]
        id_local[i] = id_local_paramRVE[i]
        
        print("Solving snapshot", int(id_[i]), int(id_local[i]), i)

        if(createMesh):
            buildRVEmesh(paramRVEdata[i,:,:], meshname, 
                         isOrdered = False, size = 'full',  NxL = 2, NyL = 2, maxOffset = 2)
        
        solve_snapshot(i, meshname, paramMaterial, opModel, snapshots, bcsDNN )
    
        
        fsnaps.flush()
        
                
    fsnaps.close()

if __name__ == '__main__':
    
    if(len(sys.argv)>1):
        run = int(sys.argv[1])
        num_runs = int(sys.argv[2])
    else:
        comm = MPI.COMM_WORLD
        run = comm.Get_rank()
        num_runs = comm.Get_size()
        
    print('run, num_runs ', run, num_runs)
    
    i0 = 0
    run = comm.Get_rank()
    num_runs = comm.Get_size()
    
    folder = rootDataPath + "/CFM/dataset/"
    
    suffix = ""
    opModel = 'per'
    createMesh = True
    
    contrast = 10.0
    E2 = 1.0
    nu = 0.3
    paramMaterial = [nu,E2*contrast,nu,E2]
    
    paramRVEname = folder +  'paramRVEdataset_subdomains{0}.hd5'.format(suffix)
    snapshotsname = folder +  'snapshots_subdomains_HF.hd5'.format(suffix,run)
    meshname = folder + "meshes/mesh_temp_subdomains_full_HF_{0}.xdmf".format(run)
    
    filesnames = [paramRVEname, snapshotsname, meshname, folder]
    
    # generation of the lite mesh for the internal boundary
    # p = paramRVE_default()
    # meshRef = degeneratedBoundaryRectangleMesh(x0 = p.x0L, y0 = p.y0L, Lx = p.LxL , Ly = p.LyL , Nb = 21)
    # meshRef.generate()
    # meshRef.write(bndMeshname , 'fenics')
    
    
    buildSnapshots(paramMaterial, filesnames, opModel, createMesh, run, num_runs, i0)
