# You should run cook.py with:
# 1. the suitable tangents.hd5 file (for the chosen bc model (caseType) ) 
# 2. macroscale mesh refinement indicator (Ny), 
# 3. A random seed to indicate how tangents.hd5 file will be shuffled (cell indexes will be matched randonmly to the RVE indexes of tangent.hd5).

import sys, os
import dolfin as df 
import matplotlib.pyplot as plt
import numpy as np
from ufl import nabla_div
from timeit import default_timer as timer
import multiphenics as mp
from mpi4py import MPI

from deepBND.__init__ import *
from deepBND.core.elasticity.fenics_utils import symgrad_voigt
from deepBND.core.fenics_tools.wrapper_solvers import solver_iterative
from deepBND.core.multiscale.misc import Chom_multiscale
import deepBND.core.data_manipulation.wrapper_h5py as myhd
from deepBND.core.fenics_tools.enriched_mesh import EnrichedMesh 
import deepBND.core.fenics_tools.wrapper_io as iofe
import deepBND.core.fenics_tools.misc as feut

from deepBND.examples.cook.mesh import CookMembrane

comm = MPI.COMM_WORLD
comm_self = MPI.COMM_SELF
rank = comm.Get_rank()
num_ranks = comm.Get_size()

def solve_cook(meshfile, tangent_dataset):

    mesh = EnrichedMesh(meshfile)

    Uh = df.VectorFunctionSpace(mesh, "CG", 2)
    clampedBndFlag = 2 
    LoadBndFlag = 1 
    
    ty = 0.05
    traction = df.Constant((0.0,ty ))
    bcL = df.DirichletBC(Uh, df.Constant((0.0,0.0)), mesh.boundaries, clampedBndFlag)
    
    mapping = np.random.randint(0,len(tangent_dataset),mesh.num_cells())
        
    Chom = Chom_multiscale(tangent_dataset, mapping, degree = 0)
    
    # # Define variational problem
    uh = df.TrialFunction(Uh) 
    vh = df.TestFunction(Uh)
    a = df.inner(df.dot(Chom,symgrad_voigt(uh)), symgrad_voigt(vh))*mesh.dx
    b = df.inner(traction,vh)*mesh.ds(LoadBndFlag)
    
    # Compute solution
    uh = solver_iterative(a, b, bcL, Uh)
    
    return uh, Chom


if __name__ == '__main__':
    
    # =========== argument input =================
    if(len(sys.argv)>1):
        Ny_split = int(sys.argv[1])
        caseType = sys.argv[2]
        seed = int(sys.argv[3])
    
    else:     
        Ny_split =  40 # 5, 10, 20, 40, 80
        caseType = 'reduced_per' # opt: reduced_per, dnn, full
        seed = 1
        
    
    # ========== dataset folders ================= 
    folder = rootDataPath + "/new_fe2/DNS/DNS_72_2/"
    folderMesh = rootDataPath + '/deepBND/cook/meshes_seed{0}/'.format(seed)
    
    tangentName = folder + 'tangents/tangent_%s.hd5'%caseType
    tangent_dataset = myhd.loadhd5(tangentName, 'tangent')
    # ids = myhd.loadhd5(tangentName, 'id') # apparenttly no need
    # center = myhd.loadhd5(tangentName, 'center') # apparenttly no need
    meshfile = folderMesh + 'mesh_%d.xdmf'%Ny_split
    
    lcar = 44.0/Ny_split  
    
    gmshMesh = CookMembrane(lcar = lcar)
    gmshMesh.write(savefile = meshfile, opt = 'fenics')
    
    np.random.seed(seed)
    
    uh, Chom = solve_cook(meshfile, tangent_dataset)
    

    sigma = lambda u: df.dot(Chom, symgrad_voigt(u))
    
    
    with df.XDMFFile(comm, folderMesh + "cook_%s_%d_vtk.xdmf"%(caseType,Ny_split)) as file:
        iofe.export_XDMF_displacement_sigma(uh, sigma, file)
    
    with df.XDMFFile(comm, folderMesh + "cook_%s_%d.xdmf"%(caseType,Ny_split)) as file:
        iofe.export_checkpoint_XDMF_displacement_sigma(uh, sigma, file)







