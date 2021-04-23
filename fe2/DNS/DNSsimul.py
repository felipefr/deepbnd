import sys, os
from dolfin import *
import numpy as np
sys.path.insert(0,'../../utils/')
import matplotlib.pyplot as plt
from ufl import nabla_div
from fenicsUtils import symgrad
import meshUtils as meut
import elasticity_utils as elut

Lx = 2.0
Ly = 0.5
ty = -0.01

# Create mesh and define function space
mesh = meut.EnrichedMesh('DNS.xdmf')
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
a = inner(sigma(uh), grad(vh))*mesh.dx
b = inner(traction,vh)*mesh.ds(3) # 3 is right face

# Compute solution
uh = Function(Uh)
solve(a == b, uh, bcs = bcL, solver_parameters={"linear_solver": "superlu"}) # normally the best for single process
# solve(a == b, uh, bcs = bcL, solver_parameters={"linear_solver": "mumps"}) # best for distributed 

print(uh.vector().get_local()[:].shape)
print(np.linalg.norm(uh.vector().get_local()[:]))

# Save solution in VTK format
fileResults = XDMFFile("barMacro.xdmf")
fileResults.parameters["flush_output"] = True
fileResults.parameters["functions_share_mesh"] = True

uh.rename("u", "label")
lame0_h = project(lame[0], FunctionSpace(mesh, 'DG', 0))
lame0_h.rename("lame0", "label")

fileResults.write(uh,0)
fileResults.write(lame0_h,0)
