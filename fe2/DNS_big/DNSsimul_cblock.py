
import sys, os
from dolfin import *
import numpy as np
sys.path.insert(0,'../../utils/')
import matplotlib.pyplot as plt
from ufl import nabla_div
from fenicsUtils import symgrad
import meshUtils as meut
import elasticity_utils as elut
from block import *
from block.iterative import *
from block.algebraic.petsc import AMG, collapse, ASM, Jacobi, ML
# from block.dolfin_util import *

from mpi4py import MPI
from timeit import default_timer as timer


comm = MPI.COMM_WORLD
comm_self = MPI.COMM_SELF
rank = comm.Get_rank()
num_ranks = comm.Get_size()

Lx = 2.0
Ly = 0.5
ty = -0.01
Ny = '48'

start = timer()
# Create mesh and define function space
def getAF(meshName, comm):
    mesh = meut.EnrichedMesh(meshName, comm)
    Uh = VectorFunctionSpace(mesh, "CG", 1)
    
    bcL = DirichletBC(Uh, Constant((0.0,0.0)), mesh.boundaries, 5) # 5 is left face
    
    # ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
    traction = Constant((0.0,ty ))
    
    contrast = 10.0
    E2 = 1.0
    nu = 0.3
    param = [nu,E2*contrast,nu,E2]
    lame = elut.getLameInclusions(*param, mesh)
    
    def sigma(u):
        return lame[0]*nabla_div(u)*Identity(2) + 2*lame[1]*symgrad(u)
    
    # Define variational problem
    uh = TrialFunction(Uh) 
    vh = TestFunction(Uh)
    a = inner(sigma(uh), symgrad(vh))*mesh.dx
    b = inner(traction,vh)*mesh.ds(3) # 3 is right face
    
    print(Uh.dim())
    
    A, F = assemble_system(a, b, bcL)
    
    return A, F, Uh

A, F, Uh = getAF('./DNS_{0}/mesh.xdmf'.format(Ny), comm)

uh = Function(Uh)
Ap = AMG(A)
# Ap = ML(A)
# Ap = AMG(A)
# Ap = AMG(A)
# Ap = AMG(A)
# Ap = ASM(A)

# AAinv = MinRes(A, precond=Ap, show=2, name='AA^')
AAinv = BiCGStab(A, precond=Ap, show=2, name='AA^')
# AAinv = SymmLQ(A, precond=Ap, show=2, name='AA^')
# AAinv = Richardson(A, precond=Ap, show=2, name='AA^')
# AAinv = MinRes2(A, precond=Ap, show=2, name='AA^')
# AAinv = TFQMR(A, precond=Ap, show=2, name='AA^')
# AAinv = LGMRES(A, precond=Ap, maxiter = 100, show=2, name='AA^')
uh.vector().set_local(AAinv * F)

end = timer()

with XDMFFile(comm, "./DNS_{0}/barMacro_DNS_P1_playingCblock.xdmf".format(Ny)) as file:
    file.write_checkpoint(uh,'u',0)

print('\n solved \n', end - start, '\n')