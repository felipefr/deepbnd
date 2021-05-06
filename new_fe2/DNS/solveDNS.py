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

f = open("../../../rootDataPath.txt")
rootData = f.read()[:-1]
f.close()

Lx = 2.0
Ly = 0.5
ty = -0.01
Ny = 24
suffix = int(sys.argv[2]) if len(sys.argv)>2 else ''

folder = rootData + '/new_fe2/DNS/DNS_%d%s/'%(Ny,suffix)

start = timer()
# Create mesh and define function space
mesh = meut.EnrichedMesh(folder + 'mesh.xdmf', comm)
Uh = VectorFunctionSpace(mesh, "CG", 2)

if(num_ranks == 1):
    V0 = assemble(Constant(1.0)*mesh.dx(0))
    V1 = assemble(Constant(1.0)*mesh.dx(1))
    print("sanitity check: ", V0/(V1+V0), V1 + V0)
    
    input()

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

uh = Function(Uh)

print(Uh.dim())

# solver.solve()
A, F = assemble_system(a, b, bcL)

solver = PETScKrylovSolver('gmres','hypre_amg')
solver.parameters["relative_tolerance"] = 1e-5
solver.parameters["absolute_tolerance"] = 1e-6
# solver.parameters["nonzero_initial_guess"] = True
solver.parameters["error_on_nonconvergence"] = False
solver.parameters["maximum_iterations"] = 1000
solver.parameters["monitor_convergence"] = True
# solver.parameters["report"] = True
# solver.parameters["preconditioner"]["ilu"]["fill_level"] = 1 # 
solver.set_operator(A)
solver.solve(uh.vector(), F)    

with XDMFFile(comm, folder + "barMacro_DNS.xdmf".format(Ny)) as file:
    file.write_checkpoint(uh,'u',0)

end = timer()

print('\n solved \n', end - start, '\n')