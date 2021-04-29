
import sys, os
import dolfin as df
import multiphenics as mp
import numpy as np
sys.path.insert(0,'../../utils/')
import matplotlib.pyplot as plt
from ufl import nabla_div
from fenicsUtils import symgrad
import meshUtils as meut
import elasticity_utils as elut

from mpi4py import MPI
from timeit import default_timer as timer


comm = MPI.COMM_WORLD
comm_self = MPI.COMM_SELF
rank = comm.Get_rank()
num_ranks = comm.Get_size()

Lx = 2.0
Ly = 0.5
ty = -0.01
Ny = '12'

start = timer()
# Create mesh and define function space
mesh = meut.EnrichedMesh('./DNS_{0}/mesh.xdmf'.format(Ny), comm)
Uh = df.VectorFunctionSpace(mesh, "CG", 1)

# bcL = DirichletBC(Uh, Constant((0.0,0.0)), mesh.boundaries, 5) # 5 is left face

# onBoundary = df.CompiledSubDomain('on_boundary')
leftBoundary = df.CompiledSubDomain('on_boundary && near(x[0], 0.0, tol)', tol=1E-14)

Wh = mp.BlockFunctionSpace([Uh, Uh], restrict = [None, leftBoundary] )

up = mp.BlockTrialFunction(Wh)
vq = mp.BlockTestFunction(Wh)
up_ = mp.block_split(up)
vq_ = mp.block_split(vq)

traction = df.Constant((0.0,ty ))

contrast = 10.0
E2 = 1.0
nu = 0.3
param = [nu,E2*contrast,nu,E2]
lame = elut.getLameInclusions(*param, mesh)

def sigma(u):
    return lame[0]*nabla_div(u)*df.Identity(2) + 2*lame[1]*symgrad(u)

# Define variational problem
uh, ph = up_
vh, qh = vq_ 


aa = [[df.inner(sigma(uh), symgrad(vh))*mesh.dx , df.inner(ph,vh)*mesh.ds(5)], [df.inner(qh,uh)*mesh.ds(5) , 0]]
bb = [df.inner(traction,vh)*mesh.ds(3),0]


A = mp.block_assemble(aa)
F = mp.block_assemble(bb)

# solver = df.PETScLUSolver("mumps") 
solver = df.PETScKrylovSolver('gmres','hypre_euclid')
# solver.set_operator(A)

# solver.parameters["linear_solver"] = 'gmres'
# solver.parameters["preconditioner"] = "hypre_amg"
# solver.parameters["krylov_solver"]["relative_tolerance"] = 1e-5
# solver.parameters["krylov_solver"]["absolute_tolerance"] = 1e-6
# solver.parameters["krylov_solver"]["nonzero_initial_guess"] = True
# solver.parameters["krylov_solver"]["error_on_nonconvergence"] = False
# solver.parameters["krylov_solver"]["maximum_iterations"] = 15
solver.parameters["monitor_convergence"] = True
solver.parameters["report"] = True
        
sol = mp.BlockFunction(Wh)
solver.solve(A, sol.block_vector(), F)    
sol.block_vector().block_function().apply("to subfunctions")


with df.XDMFFile(comm, "./DNS_{0}/barMacro_DNS_mp_iterative.xdmf".format(Ny)) as file:
    file.write_checkpoint(sol[0],'u',0)

end = timer()

print('\n solved \n', end - start, '\n')