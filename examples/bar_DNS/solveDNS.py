'''
2. Run solveDNS.py with: 
- Enter mesh, output files, Ny and some other suffixes for identification of files purposes (as used in the mesh generation). 
'''

import sys, os
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from ufl import nabla_div

from deepBND.__init__ import *
from deepBND.core.fenics_tools.misc import symgrad
from deepBND.core.fenics_tools.enriched_mesh import EnrichedMesh 
from deepBND.core.elasticity.fenics_utils import getLameInclusions
from deepBND.core.fenics_tools.wrapper_solvers import solver_iterative

from mpi4py import MPI
from timeit import default_timer as timer

comm = MPI.COMM_WORLD
comm_self = MPI.COMM_SELF
rank = comm.Get_rank()
num_ranks = comm.Get_size()

def solve_DNS(meshfile, param):
    # Create mesh and define function space
    mesh = EnrichedMesh(meshfile, comm)
    Uh = VectorFunctionSpace(mesh, "CG", 2)  
    
    FaceTraction, tx, ty, FacesClamped = param
    
    bcL = [ DirichletBC(Uh, Constant((0.0,0.0)), mesh.boundaries, i) 
                                                       for i in FacesClamped] 
    
    traction = Constant((tx,ty))
    
    contrast = 10.0
    E2 = 1.0
    nu = 0.3
    param = [nu,E2*contrast,nu,E2]
    lame = getLameInclusions(*param, mesh)
    
    def sigma(u):
        return lame[0]*nabla_div(u)*Identity(2) + 2*lame[1]*symgrad(u)
    
    # Define variational problem
    uh = TrialFunction(Uh) 
    vh = TestFunction(Uh)
    a = inner(sigma(uh), symgrad(vh))*mesh.dx
    b = inner(traction,vh)*mesh.ds(FaceTraction) # 3 is right face
    
    uh = solver_iterative(a, b, bcL, Uh)
    
    return uh
    
if __name__ == '__main__':
    
    # =========== argument input =================
    if(len(sys.argv)>1):
        Ny = int(sys.argv[1])
        suffix = sys.argv[2] if len(sys.argv)>2 else ''
    
    else:     
        Ny =  24 # 24 or 72
        suffix = '' # opt: reduced_per, dnn, full
        
    folder = rootDataPath + '/deepBND/DNS/DNS_%d%s/'%(Ny,suffix)
    meshname = folder + 'mesh.xdmf'
    outputfile = folder + "barMacro_DNS%s.xdmf"%suffix

    # Lx = 2.0
    # Ly = 0.5
    FaceTraction = 3    
    ty = -0.01
    tx = 0.0    
    FacesClamped = [5]
        
    param = [FaceTraction, tx, ty, FacesClamped]
        
    start = timer()
    
    uh = solve_DNS(meshname, param)
    
    with XDMFFile(comm, outputfile) as file:
        file.write_checkpoint(uh,'u',0)
    
    end = timer()
    
    print('\n solved \n', end - start, '\n')
    