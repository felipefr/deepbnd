#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 04:46:39 2022

@author: felipe
"""

import sys, os
import numpy as np
from numpy import isclose
from dolfin import *
from multiphenics import *
import copy
from timeit import default_timer as timer
from mpi4py import MPI

from deepBND.__init__ import *
from deepBND.core.multiscale.mesh_RVE import buildRVEmesh
from deepBND.core.multiscale.micro_model_gen import MicroConstitutiveModelGen
import deepBND.core.elasticity.fenics_utils as fela
import deepBND.core.fenics_tools.wrapper_io as iofe
import deepBND.core.multiscale.misc as mtsm
from deepBND.core.fenics_tools.enriched_mesh import EnrichedMesh 
import deepBND.core.data_manipulation.wrapper_h5py as myhd
from deepBND.core.mesh.degenerated_rectangle_mesh import degeneratedBoundaryRectangleMesh
from deepBND.core.multiscale.mesh_RVE import paramRVE_default

# from deepBND.creation_model.dataset.simulation_snapshots import *

comm = MPI.COMM_WORLD
comm_self = MPI.COMM_SELF

# comm = MPI.COMM_WORLD
# comm_self = MPI.COMM_SELF

totalTime = 0.0 

def solve_snapshot(i, meshname, paramMaterial, opModel, datasets):
    
    global totalTime
    
    tangent, tangentL, sigma, sigmaL, eps, epsL = datasets[1:]
    
    start = timer()
    
    microModel = MicroConstitutiveModelGen(meshname, paramMaterial, opModel)
      
    Hom = microModel.computeHomogenisation([0,1])   
    tangent[i,:,:] = Hom['tangent']
    tangentL[i,:,:] = Hom['tangentL']
    sigma[i,:,:] = Hom['sigma']
    sigmaL[i,:,:] = Hom['sigmaL']
    eps[i,:,:] = Hom['eps']
    epsL[i,:,:] = Hom['epsL']
    
    end = timer()
    
    totalTime = totalTime + (end - start)
    print("concluded in ", end - start)      
    

def buildSnapshots(paramMaterial, filesnames, opModel, createMesh, run, num_runs, i0, comm_self):
    paramRVEname, snapshotsname, meshname = filesnames
    
    ns = len(myhd.loadhd5(paramRVEname, 'param')[:,0])
    
    ids_run = np.arange(run,ns,num_runs).astype('int')
    nrun = len(ids_run)
    
    os.system('rm ' + snapshotsname)
    snapshots, fsnaps = myhd.zeros_openFile(filename = snapshotsname,  
                                            shape = [(nrun,)] + 6*[(nrun,3,3)],
                                            label = ['id', 'tangent', 'tangentL', 'sigma', 'sigmaL', 'eps', 'epsL'], mode = 'w-')
    
    ids = snapshots[0]
             
    paramRVEdata = myhd.loadhd5(paramRVEname, 'param')[ids_run] 
        
    for i in range(i0,nrun):
        
        ids[i] = ids_run[i]
        
        print("Solving snapshot", int(ids[i]), i)

        if(createMesh):
            buildRVEmesh(paramRVEdata[i,:,:], meshname, 
                         isOrdered = False, size = 'full',  NxL = 6, NyL = 6, maxOffset = 2)
        
        solve_snapshot(i, meshname, paramMaterial, opModel, snapshots)
    
        
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
    
    suffix = "_ns100"
    opModel = 'per'
    createMesh = True
    
    contrast = 10.0
    E2 = 1.0
    nu = 0.3
    paramMaterial = [nu,E2*contrast,nu,E2]
    
    paramRVEname = folder +  'paramRVEdataset{0}.hd5'.format(suffix)
    snapshotsname = folder +  'snapshots{0}_full.hd5'.format(suffix)
    meshname = folder + "meshes/mesh_temp_{0}.xdmf".format(run)
    
    filesnames = [paramRVEname, snapshotsname, meshname]
    
    # generation of the lite mesh for the internal boundary
    # p = paramRVE_default()
    # meshRef = degeneratedBoundaryRectangleMesh(x0 = p.x0L, y0 = p.y0L, Lx = p.LxL , Ly = p.LyL , Nb = 21)
    # meshRef.generate()
    # meshRef.write(bndMeshname , 'fenics')
    
    
    buildSnapshots(paramMaterial, filesnames, opModel, createMesh, run, num_runs, i0, comm_self)
    
    print(totalTime)
