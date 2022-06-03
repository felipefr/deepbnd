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
import deepBND.core.multiscale.micro_model as mscm
from deepBND.core.multiscale.mesh_RVE import paramRVE_default
import deepBND.core.multiscale.misc as mtsm

from fetricks.fenics.mesh.mesh import Mesh 
import fetricks.fenics.postprocessing.wrapper_io as iofe
import fetricks.data_manipulation.wrapper_h5py as myhd
from fetricks.fenics.mesh.degenerated_rectangle_mesh import degeneratedBoundaryRectangleMesh
from joblib import Parallel, delayed

# from dask.distributed import Client
# client = Client(n_workers=4)
# from dask import delayed



def wrap_solve_snapshot(i, paramRVEname, bndMeshname, snapshotsname, opModel):
    
        
    Mref = Mesh(bndMeshname)
    Vref = VectorFunctionSpace(Mref,"CG", 2)
    usol = Function(Vref)
                                                 
    ids = np.zeros(1)
    
    sol_S = np.zeros(Vref.dim())  
    sigma_S = np.zeros(3) 
    a_S = np.zeros(2) 
    B_S = np.zeros((2,2))
    sigmaT_S = np.zeros(3)
    
    sol_A = np.zeros(Vref.dim())  
    sigma_A = np.zeros(3) 
    a_A = np.zeros(2) 
    B_A = np.zeros((2,2))
    sigmaT_A = np.zeros(3)
        
    label = ['id', 'solutions_S','sigma_S','a_S','B_S', 'sigmaTotal_S',
             'solutions_A','sigma_A','a_A','B_A', 'sigmaTotal_A']
    
    fields = [ids, sol_S, sigma_S, a_S, B_S, sigmaT_S, sol_A, sigma_A, a_A, B_A, sigmaT_A]
    
    
    ids, sol_S, sigma_S, a_S, B_S, sigmaT_S, sol_A, sigma_A, a_A, B_A, sigmaT_A = fields
    
    paramRVEdata = myhd.loadhd5(paramRVEname, 'param')[i]
    ids[0] = myhd.loadhd5(paramRVEname, 'ids')[i]

    print("Solving snapshot", int(ids[0]), i)

    meshname_i = meshname.format(i)
    solve_snapshot(meshname_i, paramRVEdata, paramMaterial, opModel, fields[1:], usol)

    myhd.savehd5(snapshotsname.format(ids[0]), fields, label, 'w')
    
    for d in fields:
        d.fill(0.0)


def solve_snapshot(meshname, paramRVEdata, paramMaterial, opModel, datasets, usol):
    sol_S, sigma_S, a_S, B_S, sigmaT_S, sol_A, sigma_A, a_A, B_A, sigmaT_A = datasets
    
    start = timer()

    buildRVEmesh(paramRVEdata, meshname, isOrdered = False, size = 'full', NxL = 4, NyL = 4, maxOffset = 2, lcar = 3/30)
        
    microModel = mscm.MicroModel(meshname, paramMaterial, opModel)
    microModel.compute()
    
    # meshname_postproc = meshname[:-5] + '_to_the_paper.xdmf'
    # microModel.visualiseMicrostructure(meshname_postproc)
    
    for j_voigt, sol, sigma, a, B, sigmaT in zip([2,0], [sol_S,sol_A],[sigma_S,sigma_A],
                                                 [a_S,a_A],[B_S,B_A],[sigmaT_S, sigmaT_A]):
        
        T, a[:], B[:,:] = mtsm.getAffineTransformationLocal(microModel.sol[j_voigt][0], 
                                                                microModel.mesh, [0,1], justTranslation = False)    
    
        usol.interpolate(microModel.sol[j_voigt][0])
       
        sol[:] = usol.vector().get_local()[:]
        sigma[:] = microModel.homogenise([0,1],j_voigt).flatten()[[0,3,2]]    
        sigmaT[:] = microModel.homogenise([0,1,2,3],j_voigt).flatten()[[0,3,2]]       
    
    end = timer()

    print("concluded in ", end - start)      
    

def buildSnapshots(paramMaterial, filesnames, opModel, createMesh):
    bndMeshname, paramRVEname, snapshotsname, meshname = filesnames
    

    # ns = len(myhd.loadhd5(paramRVEname, 'ids'))

    Parallel(n_jobs=2)(delayed(wrap_solve_snapshot)(i, paramRVEname, bndMeshname, snapshotsname, opModel) for i in range(10))

if __name__ == '__main__':
    

    folder = rootDataPath + "/review2/dataset/"
    
    suffix = ""
    opModel = 'per'
    createMesh = True
    
    contrast = 10.0
    E2 = 1.0
    nu = 0.3
    paramMaterial = [nu,E2*contrast,nu,E2]
    
    bndMeshname = folder + 'boundaryMesh.xdmf'
    paramRVEname = folder +  'paramRVEdataset{0}.hd5'.format(suffix)
    snapshotsname = folder +  'snapshots_tests_parallel.hd5'
    meshname = folder + "meshes/mesh_temp_{0}.xdmf"
    
    run = int(sys.argv[1])
    numruns = int(sys.argv[2])
    
    if(run == -1): # preparation paramRVE (splitting)
        
        ns = len(myhd.loadhd5(paramRVEname, 'ids'))
        
        size = int(np.floor(ns/numruns)) 
    
        labels = ['ids', 'param']
        
        indexesOutput = [ np.arange(i*size, (i+1)*size) for i in range(numruns)]
        indexesOutput[-1] = np.arange((numruns-1)*size, ns) # in case the division is not exact
        
        myhd.split(paramRVEname, indexesOutput, labels, 'w')

        os.system('mkdir ' + snapshotsname.split('.')[0] + '_split/')        

        p = paramRVE_default(NxL = 4, NyL = 4, maxOffset = 4)
        meshRef = degeneratedBoundaryRectangleMesh(x0 = p.x0L, y0 = p.y0L, Lx = p.LxL , Ly = p.LyL , Nb = 100)
        meshRef.generate()
        meshRef.write(bndMeshname , 'fenics')
            
    elif(run<numruns):
        start = timer()
        if(numruns>1):
            paramRVEname = paramRVEname.split('.')[0] + '_split/part_{0}'.format(run) + '.hd5'
            snapshotsname = snapshotsname.split('.')[0] + '_split/part_{0}' + '.hd5'
            
        labels = ['id', 'param']

        # meshname = meshname
        filesnames = [bndMeshname, paramRVEname, snapshotsname, meshname]
        
        buildSnapshots(paramMaterial, filesnames, opModel, createMesh)
        end = timer()
        print("time ellapsed :" , end - start )
    else:
    
        labels =  ['id', 'solutions_S','sigma_S','a_S','B_S', 'sigmaTotal_S',
                 'solutions_A','sigma_A','a_A','B_A', 'sigmaTotal_A']
        
        snapshotsname_rad = snapshotsname.split('.')[0] + '_split/part_{0}' + '.hd5'
        
        myhd.merge([snapshotsname_rad.format(i) for i in range(numruns)], snapshotsname, labels, labels)