#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 09:07:03 2022

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

def solve_snapshot(i, meshname, paramMaterial, opModel, datasets):
    ids, tangent = datasets
    
    start = timer()
    
    microModel = MicroConstitutiveModelGen(meshname, paramMaterial, opModel)
      
    tangent[i,:,:] = microModel.computeTangent([0,1])[0]   
    
    end = timer()

    print("concluded in ", end - start)      
    

def buildSnapshots(paramMaterial, filesnames, opModel, createMesh, run, num_runs, i0, comm_self):
    paramRVEname, snapshotsname, meshname = filesnames
    
    ns = len(myhd.loadhd5(paramRVEname, 'param')[:,0])
    
    ids_run = np.arange(run,ns,num_runs).astype('int')
    nrun = len(ids_run)
    
    os.system('rm ' + snapshotsname)
    snapshots, fsnaps = myhd.zeros_openFile(filename = snapshotsname,  
                                            shape = [(nrun,)] + [(nrun,3,3)],
                                            label = ['id', 'tangent'], mode = 'w-')
    
    ids, tangent = snapshots
             
    paramRVEdata = myhd.loadhd5(paramRVEname, 'param')[ids_run] 
        
    for i in range(i0,nrun):
        
        ids[i] = ids_run[i]
        
        print("Solving snapshot", int(ids[i]), i)

        if(createMesh):
            buildRVEmesh(paramRVEdata[i,:,:], meshname, 
                         isOrdered = False, size = 'reduced',  NxL = 6, NyL = 6, maxOffset = 2)
        
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
    
    suffix = ""
    opModel = 'MR'
    createMesh = True
    
    contrast = 10.0
    E2 = 1.0
    nu = 0.3
    paramMaterial = [nu,E2*contrast,nu,E2]
    
    paramRVEname = folder +  'paramRVEdataset{0}.hd5'.format(suffix)
    snapshotsname = folder +  'snapshots_reduced_MR.hd5'.format(suffix,run)
    meshname = folder + "meshes/mesh_temp_reduced_{0}.xdmf".format(run)
    
    filesnames = [paramRVEname, snapshotsname, meshname]
    
    # generation of the lite mesh for the internal boundary
    # p = paramRVE_default()
    # meshRef = degeneratedBoundaryRectangleMesh(x0 = p.x0L, y0 = p.y0L, Lx = p.LxL , Ly = p.LyL , Nb = 21)
    # meshRef.generate()
    # meshRef.write(bndMeshname , 'fenics')
    
    
    buildSnapshots(paramMaterial, filesnames, opModel, createMesh, run, num_runs, i0, comm_self)
