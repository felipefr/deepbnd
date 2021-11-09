import sys, os
from numpy import isclose
from dolfin import *
from multiphenics import *
sys.path.insert(0, '../../utils/')

import fenicsWrapperElasticity as fela
import matplotlib.pyplot as plt
import numpy as np
import meshUtils as meut
import generationInclusions as geni
import fenicsMultiscale as fmts
import elasticity_utils as elut
import fenicsWrapperElasticity as fela
import multiphenicsMultiscale as mpms
import ioFenicsWrappers as iofe
import fenicsUtils as feut
import myHDF5 as myhd
import matplotlib.pyplot as plt
import copy
from timeit import default_timer as timer
import MicroModel as micm

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_self = MPI.COMM_SELF
#run = comm.Get_rank()
#num_runs = comm.Get_size()

run = int(sys.argv[1])
num_runs = int(sys.argv[2])

f = open("../../../rootDataPath.txt")
rootData = f.read()[:-1]
f.close()

folder = rootData + "/new_fe2/dataset/"

suffix = "_train"
opModel = 'per'
createMesh = True

contrast = 10.0
E2 = 1.0
nu = 0.3
paramMaterial = [nu,E2*contrast,nu,E2]

p = geni.paramRVE()

# meshRef = meut.degeneratedBoundaryRectangleMesh(x0 = p.x0L, y0 = p.y0L, Lx = p.LxL , Ly = p.LyL , Nb = 21)
# meshRef.generate()
# meshRef.write(folder + 'boundaryMesh.xdmf', 'fenics')
Mref = meut.EnrichedMesh(folder + 'boundaryMesh.xdmf',comm = comm_self)
Vref = VectorFunctionSpace(Mref,"CG", 1)
usol = Function(Vref)

ns = len(myhd.loadhd5(folder +  'paramRVEdataset{0}.hd5'.format(suffix), 'param')[:,0])

# ns = 2
# ids_run = np.arange(run,ns,num_runs).astype('int')
ids_run = np.arange(run,ns,num_runs).astype('int')
nrun = len(ids_run)

os.system('rm ' + folder +  'snapshots{0}_{1}.h5'.format(suffix,run))
snapshots, fsnaps = myhd.zeros_openFile(filename = folder +  'snapshots{0}_{1}.h5'.format(suffix,run),  
                                        shape = [(nrun,)] + 2*[(nrun,Vref.dim()),(nrun,3),(nrun,2), (nrun,2,2), (nrun,3)],
                                        label = ['id', 'solutions_S','sigma_S','a_S','B_S', 'sigmaTotal_S',
                                                 'solutions_A','sigma_A','a_A','B_A', 'sigmaTotal_A'], mode = 'w-')

ids, sol_S, sigma_S, a_S, B_S, sigmaT_S, sol_A, sigma_A, a_A, B_A, sigmaT_A = snapshots

         
paramRVEdata = myhd.loadhd5(folder +  'paramRVEdataset{0}.hd5'.format(suffix), 'param')[ids_run] 
permTotal = geni.orderedIndexesTotal(p.Nx,p.Ny,p.NxL)

for i in range(nrun):
    ids[i] = ids_run[i]
    
    print("Solving snapshot", int(ids[i]), i)
    start = timer()
    meshGMSH = meut.ellipseMesh2Domains(p.x0L, p.y0L, p.LxL, p.LyL, p.NxL*p.NyL, paramRVEdata[i,permTotal,:], 
                                        p.Lxt, p.Lyt, p.lcar, x0 = p.x0, y0 = p.y0)
    meshGMSH.setTransfiniteBoundary(p.NpLxt)
    meshGMSH.setTransfiniteInternalBoundary(p.NpLxL)   
        
    meshGMSH.write(folder + "meshes{0}/mesh_temp_{1}.xdmf".format(suffix,run), opt = 'fenics')
    
    microModel = micm.MicroModel(folder + "meshes{0}/mesh_temp_{1}.xdmf".format(suffix,run), paramMaterial, opModel)
    
    microModel.compute()
    
    for j_voigt, sol, sigma, a, B, sigmaT in zip([2,0], [sol_S,sol_A],[sigma_S,sigma_A],
                                                 [a_S,a_A],[B_S,B_A],[sigmaT_S, sigmaT_A]):
        
        T, a[i,:], B[i,:,:] = feut.getAffineTransformationLocal(microModel.sol[j_voigt][0], microModel.mesh, [0,1], justTranslation = False)    
    
        usol.interpolate(microModel.sol[j_voigt][0])
       
        sol[i,:] = usol.vector().get_local()[:]
        sigma[i,:] = microModel.homogenise([0,1],j_voigt).flatten()[[0,3,2]]    
        sigmaT[i,:] = microModel.homogenise([0,1,2,3],j_voigt).flatten()[[0,3,2]]       
    
    end = timer()

    print("concluded in ", end - start)      
    fsnaps.flush()
    
            
fsnaps.close()


