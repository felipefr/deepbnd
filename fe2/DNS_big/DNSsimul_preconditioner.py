
import sys, os
from dolfin import *
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
Ny = '48'
Ny0 = '6'


# uh0 = Expression(("0.0","-A*x[0]*x[0]"), A = 0.01, degree = 1)

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
    
    # solver.solve()
    A, F = assemble_system(a, b, bcL)
    
    return A, F, Uh

A, F, Uh = getAF('./DNS_{0}/mesh.xdmf'.format(Ny), comm)

uh = Function(Uh)
# uh.interpolate(uh0)  

uh.vector().set_local(np.random.rand(Uh.dim()))

def getP(Uh):
    
    def sigma(u):
        return nabla_div(u)*Identity(2)
        
    # Define variational problem
    uh = TrialFunction(Uh) 
    vh = TestFunction(Uh)
    a = inner(sigma(uh), symgrad(vh))*Uh.mesh().dx
    
    
    # solver.solve()
    P = assemble(a)
    
    return P


# P = getP(Uh)


solver = PETScKrylovSolver('bicgstab','hypre_amg')
solver.parameters["relative_tolerance"] = 1e-5
solver.parameters["absolute_tolerance"] = 1e-6
# solver.parameters["nonzero_initial_guess"] = True
solver.parameters["error_on_nonconvergence"] = False
solver.parameters["maximum_iterations"] = 1000
solver.parameters["monitor_convergence"] = True
solver.parameters["report"] = True
# solver.parameters["preconditioner"]["ilu"]["fill_level"] = 1 # 
solver.set_operator(A)
solver.solve(uh.vector(), F)    

# print("L2 norm", assemble(inner(uh,uh)*Uh.mesh().dx))

with XDMFFile(comm, "./DNS_{0}/barMacro_DNS_P1_petsc.xdmf".format(Ny)) as file:
    file.write_checkpoint(uh,'u',0)

end = timer()

print('\n solved \n', end - start, '\n')