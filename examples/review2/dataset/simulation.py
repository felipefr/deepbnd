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


def solve_snapshot(i, meshname, paramMaterial, opModel, datasets, usol):
    ids, sol_S, sigma_S, a_S, B_S, sigmaT_S, sol_A, sigma_A, a_A, B_A, sigmaT_A = datasets
    
    start = timer()
    
    microModel = mscm.MicroModel(meshname, paramMaterial, opModel)
    microModel.compute()
    
    # meshname_postproc = meshname[:-5] + '_to_the_paper.xdmf'
    # microModel.visualiseMicrostructure(meshname_postproc)
    
    for j_voigt, sol, sigma, a, B, sigmaT in zip([2,0], [sol_S,sol_A],[sigma_S,sigma_A],
                                                 [a_S,a_A],[B_S,B_A],[sigmaT_S, sigmaT_A]):
        
        T, a[i,:], B[i,:,:] = mtsm.getAffineTransformationLocal(microModel.sol[j_voigt][0], 
                                                                microModel.mesh, [0,1], justTranslation = False)    
    
        usol.interpolate(microModel.sol[j_voigt][0])
       
        sol[i,:] = usol.vector().get_local()[:]
        sigma[i,:] = microModel.homogenise([0,1],j_voigt).flatten()[[0,3,2]]    
        sigmaT[i,:] = microModel.homogenise([0,1,2,3],j_voigt).flatten()[[0,3,2]]       
    
    end = timer()

    print("concluded in ", end - start)      
    

def buildSnapshots(paramMaterial, filesnames, opModel, createMesh):
    bndMeshname, paramRVEname, snapshotsname, meshname = filesnames
    
    Mref = Mesh(bndMeshname)
    Vref = VectorFunctionSpace(Mref,"CG", 2)
    usol = Function(Vref)
    
    
    paramRVEdata = myhd.loadhd5(paramRVEname, 'param') 
    ids_param = myhd.loadhd5(paramRVEname, 'ids')
    ns = len(ids_param)
    
    os.system('rm ' + snapshotsname)
    snapshots, fsnaps = myhd.zeros_openFile(filename = snapshotsname,  
                                            shape = [(ns,)] + 2*[(ns,Vref.dim()),(ns,3),(ns,2), (ns,2,2), (ns,3)],
                                            label = ['id', 'solutions_S','sigma_S','a_S','B_S', 'sigmaTotal_S',
                                                     'solutions_A','sigma_A','a_A','B_A', 'sigmaTotal_A'], mode = 'w-')
    
    ids, sol_S, sigma_S, a_S, B_S, sigmaT_S, sol_A, sigma_A, a_A, B_A, sigmaT_A = snapshots
    
             
    
    for i in range(ns):

        ids[i] = ids_param[i]
        
        print("Solving snapshot", int(ids[i]), i)

        if(createMesh):
            buildRVEmesh(paramRVEdata[i,:,:], meshname, 
                         isOrdered = False, size = 'full', NxL = 4, NyL = 4, maxOffset = 2, lcar = 3/30)
        
        solve_snapshot(i, meshname, paramMaterial, opModel, snapshots, usol)
    
        
        fsnaps.flush()
        
                
    fsnaps.close()

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
    paramRVEname = folder +  'paramRVEdataset{0}_specific.hd5'.format(suffix)
    snapshotsname = folder +  'snapshots_specific.hd5'
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

        # p = paramRVE_default(NxL = 4, NyL = 4, maxOffset = 4)
        # meshRef = degeneratedBoundaryRectangleMesh(x0 = p.x0L, y0 = p.y0L, Lx = p.LxL , Ly = p.LyL , Nb = 100)
        # meshRef.generate()
        # meshRef.write(bndMeshname , 'fenics')
            
    elif(run == -2): # preparation paramRVE (splitting) for specific indexes

        export_indices = np.concatenate((np.arange(31050,36000), np.arange(37048,42000), 
                                   np.arange(43053,48000), np.arange(49044,54000), np.arange(55032,60000))).flatten()

        
        ns = len(export_indices)        
        size = int(np.floor(ns/numruns))
        
        
        labels = ['ids', 'param']
        
        indexesOutput = [ export_indices[i*size: (i+1)*size] for i in range(numruns)]
        indexesOutput[-1] = export_indices[(numruns - 1)*size: ns] # in case the division is not exact
        
        myhd.split(paramRVEname, indexesOutput, labels, 'w')

        os.system('mkdir ' + snapshotsname.split('.')[0] + '_split')        


    elif(run>numruns):
        if(numruns>1):
            paramRVEname = paramRVEname.split('.')[0] + '_split/part_{0}'.format(run) + '.hd5'
            snapshotsname = snapshotsname.split('.')[0] + '_split/part_{0}'.format(run) + '.hd5'
            
        ns = len(myhd.loadhd5(paramRVEname, 'ids'))
    
        labels = ['id', 'param']

        meshname = meshname.format(run)
        filesnames = [bndMeshname, paramRVEname, snapshotsname, meshname]
        
        buildSnapshots(paramMaterial, filesnames, opModel, createMesh)
    
    else:
    
        labels =  ['id', 'solutions_S','sigma_S','a_S','B_S', 'sigmaTotal_S',
                 'solutions_A','sigma_A','a_A','B_A', 'sigmaTotal_A']
        
        snapshotsname_rad = snapshotsname.split('.')[0] + '_split/part_{0}' + '.hd5'
        
        myhd.merge([snapshotsname_rad.format(i) for i in range(numruns)], snapshotsname, labels, labels)