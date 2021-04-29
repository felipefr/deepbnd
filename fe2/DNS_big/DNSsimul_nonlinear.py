
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
Ny = '6'

# uh0 = Expression(("0.0","-A*x[0]*x[0]"), A = 0.01, degree = 1)

start = timer()

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
uh = Function(Uh) 
vh = TestFunction(Uh)
duh = TrialFunction(Uh) 
F = inner(sigma(uh), symgrad(vh))*mesh.dx - inner(traction,vh)*mesh.ds(3)

J = derivative(F,uh, duh)

wh = Function(Uh) 


# solve(F == 0, wh, bcL, J=J,
#       solver_parameters={"linear_solver": "gmres", "preconditioner": "hypre_amg"},
#       form_compiler_parameters={"optimize": True})


# solve(F == 0, wh, bcL, J=J)


problem = NonlinearVariationalProblem(F, wh, bcL, J, form_compiler_parameters={"optimize": True})
solver  = NonlinearVariationalSolver(problem)

iterative_solver = False

prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1E-8
prm['newton_solver']['relative_tolerance'] = 1E-7
prm['newton_solver']['maximum_iterations'] = 25
prm['newton_solver']['relaxation_parameter'] = 1.0
prm['newton_solver']['linear_solver'] = 'gmres'
prm['newton_solver']['preconditioner'] = 'hypre_amg'
if iterative_solver:
    prm['linear_solver'] = 'gmres'
    prm['preconditioner'] = 'ilu'
    prm['krylov_solver']['absolute_tolerance'] = 1E-9
    prm['krylov_solver']['relative_tolerance'] = 1E-7
    prm['krylov_solver']['maximum_iterations'] = 1000
    prm['krylov_solver']['gmres']['restart'] = 40
    prm['krylov_solver']['preconditioner']['ilu']['fill_level'] = 0
# set_log_level(PROGRESS)

solver.solve()

with XDMFFile(comm, "./DNS_{0}/barMacro_DNS_P1_nonlinear.xdmf".format(Ny)) as file:
    file.write_checkpoint(wh,'u',0)

end = timer()

print('\n solved \n', end - start, '\n')