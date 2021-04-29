
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

start = timer()
# Create mesh and define function space
mesh = meut.EnrichedMesh('./DNS_{0}/mesh.xdmf'.format(Ny), comm)
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

uh = Function(Uh)

print(Uh.dim())

# problem = LinearVariationalProblem(a, b, uh, bcs = bcL)
# solver = LinearVariationalSolver(problem)
# solver.parameters["linear_solver"] = 'gmres'
# solver.parameters["preconditioner"] = "hypre_amg"
# solver.parameters["krylov_solver"]["relative_tolerance"] = 1e-5
# solver.parameters["krylov_solver"]["absolute_tolerance"] = 1e-6
# solver.parameters["krylov_solver"]["nonzero_initial_guess"] = True
# solver.parameters["krylov_solver"]["error_on_nonconvergence"] = False
# solver.parameters["krylov_solver"]["maximum_iterations"] = 15
# solver.parameters["krylov_solver"]["monitor_convergence"] = True
# solver.parameters["krylov_solver"]["report"] = True
# solver.parameters["preconditioner"]["ilu"]["fill_level"] = 1 # 


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
# solver.set_operators
# solver.solve(uh.vector(), F)    


# parms = parameters["krylov_solver"]
# parms["relative_tolerance"]=1.e-1
# parms["absolute_tolerance"]=1.e-2
# parms["monitor_convergence"] = True
# parms["relative_tolerance"]=1.e-5   
# parms["absolute_tolerance"]=1.e-6
# parms["nonzero_initial_guess"] = True
# parms["error_on_nonconvergence"] = False
# parms["maximum_iterations"] = 15

# gmres_param = parms["gmres"]
# gmres_param['restart'] = 10

# solve(a == b, uh, bcs = bcL,
# solver_parameters={"linear_solver": "gmres",
# "preconditioner": "hypre_amg",
# "krylov_solver": parms})

# Compute solution
# solve(a == b, uh, bcs = bcL, solver_parameters={"linear_solver": "gmres", "preconditioner" : "hypre_amg"}) # best for distributed 


with XDMFFile(comm, "./DNS_{0}/barMacro_DNS_P1.xdmf".format(Ny)) as file:
    file.write_checkpoint(uh,'u',0)

end = timer()

print('\n solved \n', end - start, '\n')