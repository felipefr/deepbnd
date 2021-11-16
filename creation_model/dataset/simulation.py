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
import deepBND.core.elasticity.fenics_utils as fela
import deepBND.core.fenics_tools.wrapper_io as iofe
import deepBND.core.multiscale.misc as mtsm
from deepBND.core.fenics_tools.enriched_mesh import EnrichedMesh 
import deepBND.core.data_manipulation.wrapper_h5py as myhd
from deepBND.core.mesh.degenerated_rectangle_mesh import degeneratedBoundaryRectangleMesh
from deepBND.core.multiscale.mesh_RVE import paramRVE_default


comm = MPI.COMM_WORLD
comm_self = MPI.COMM_SELF


def solve_snapshot(i, meshname, paramMaterial, opModel, datasets, usol):
    ids, sol_S, sigma_S, a_S, B_S, sigmaT_S, sol_A, sigma_A, a_A, B_A, sigmaT_A = datasets
    
    start = timer()
    
    microModel = mscm.MicroModel(meshname, paramMaterial, opModel)
    microModel.compute()

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
    
    Mref = EnrichedMesh(bndMeshname, comm = comm_self)
    Vref = VectorFunctionSpace(Mref,"CG", 1)
    usol = Function(Vref)
    
    ns = len(myhd.loadhd5(paramRVEname, 'param')[:,0])
    
    ids_run = np.arange(run,ns,num_runs).astype('int')
    nrun = len(ids_run)
    
    os.system('rm ' + snapshotsname)
    snapshots, fsnaps = myhd.zeros_openFile(filename = snapshotsname,  
                                            shape = [(nrun,)] + 2*[(nrun,Vref.dim()),(nrun,3),(nrun,2), (nrun,2,2), (nrun,3)],
                                            label = ['id', 'solutions_S','sigma_S','a_S','B_S', 'sigmaTotal_S',
                                                     'solutions_A','sigma_A','a_A','B_A', 'sigmaTotal_A'], mode = 'w-')
    
    ids, sol_S, sigma_S, a_S, B_S, sigmaT_S, sol_A, sigma_A, a_A, B_A, sigmaT_A = snapshots
    
             
    paramRVEdata = myhd.loadhd5(paramRVEname, 'param')[ids_run] 
    
    
    for i in range(nrun):
        
        ids[i] = ids_run[i]
        
        print("Solving snapshot", int(ids[i]), i)

        if(createMesh):
            buildRVEmesh(paramRVEdata[i,:,:], meshname.format(suffix,run), 
                         isOrdinated = False, size = 'full')
        
        solve_snapshot(i, meshname.format(suffix,run), paramMaterial, opModel, snapshots, usol)
    
        
        fsnaps.flush()
        
                
    fsnaps.close()

if __name__ == '__main__':
    
    run = comm.Get_rank()
    num_runs = comm.Get_size()
    
    folder = rootDataPath + "/deepBND/dataset/"

    suffix = "_github"
    opModel = 'per'
    createMesh = True
    
    contrast = 10.0
    E2 = 1.0
    nu = 0.3
    paramMaterial = [nu,E2*contrast,nu,E2]

    bndMeshname = folder + 'boundaryMesh.xdmf'
    paramRVEname = folder +  'paramRVEdataset{0}.hd5'.format(suffix)
    snapshotsname = folder +  'snapshots{0}_{1}.h5'.format(suffix,run)
    meshname = folder + "meshes/mesh_temp_{0}.xdmf".format(run)
    
    filesnames = [bndMeshname, paramRVEname, snapshotsname, meshname]
    
    # generation of the lite mesh for the internal boundary
    # p = paramRVE_default()
    # meshRef = degeneratedBoundaryRectangleMesh(x0 = p.x0L, y0 = p.y0L, Lx = p.LxL , Ly = p.LyL , Nb = 21)
    # meshRef.generate()
    # meshRef.write(bndMeshname , 'fenics')
    
    buildSnapshots(paramMaterial, filesnames, opModel, createMesh)
    
    