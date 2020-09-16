import sys
from numpy import isclose
from dolfin import *
# import matplotlib.pyplot as plt
from multiphenics import *
sys.path.insert(0, '../utils/')

import fenicsWrapperElasticity as fela
import matplotlib.pyplot as plt
import numpy as np

# extend to vectorial
# put zero-mean case
# change mesh

mesh = fela.EnrichedMesh("data/circle.xml")

class OnBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

onBoundary = OnBoundary()
boundary_restriction = MeshRestriction(mesh, onBoundary)

V = FunctionSpace(mesh, "Lagrange", 2)

g = Expression("sin(3*x[0] + 1)*sin(3*x[1] + 1)", element=V.ufl_element())

# Solving with Multiphenics
W = BlockFunctionSpace([V, V], restrict=[None, boundary_restriction])

ul = BlockTrialFunction(W)
(u, l) = block_split(ul)
vm = BlockTestFunction(W)
(v, m) = block_split(vm)

a = [[inner(grad(u), grad(v))*dx, l*v*ds],
     [u*m*ds                    , 0     ]]
f =  [v*dx                      , g*m*ds]

A = block_assemble(a)
F = block_assemble(f)

U = BlockFunction(W)
block_solve(A, U.block_vector(), F)

plt.figure(1)
plot(U[0])
plt.figure(2)
plot(U[1])
# plt.show()

# Solving with standard Fenics
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v))*dx
f = v*dx
A = assemble(a)
F = assemble(f)
bc = DirichletBC(V, g, mesh.boundaries, 1)
bc.apply(A)
bc.apply(F)

U_ex = Function(V)

solve(A, U_ex.vector(), F)

plt.figure(3)
plot(U_ex)
plt.show()

# error

err = np.sqrt( assemble(dot(U_ex - U[0], U_ex - U[0])*dx))
print("err", err)
