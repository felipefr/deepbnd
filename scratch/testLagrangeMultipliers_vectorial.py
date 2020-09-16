import sys
from numpy import isclose
from dolfin import *
# import matplotlib.pyplot as plt
from multiphenics import *
sys.path.insert(0, '../utils/')

import fenicsWrapperElasticity as fela
import matplotlib.pyplot as plt
import numpy as np

# put zero-mean case
# change mesh

mesh = fela.EnrichedMesh("data/circle.xml")

class OnBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

onBoundary = OnBoundary()
boundary_restriction = MeshRestriction(mesh, onBoundary)

V = VectorFunctionSpace(mesh, "Lagrange", 1)

g = Expression(("sin(3*x[0] + 1)*sin(3*x[1] + 1)","cos(x[0] + 1)*cos(x[1] + 1)"), element=V.ufl_element())
# g2 = Constant(0.0)
# g = as_vector((g1,g2))

# Solving with Multiphenics
W = BlockFunctionSpace([V, V], restrict=[None, boundary_restriction])

ul = BlockTrialFunction(W)
(u, l) = block_split(ul)
vm = BlockTestFunction(W)
(v, m) = block_split(vm)

a = [[inner(grad(u), grad(v))*dx, inner(l,v)*ds],
      [inner(u,m)*ds                    , 0     ]]
f =  [ inner(v,Constant((1.0,1.0)))*dx , inner(g,m)*ds]

A = block_assemble(a)
F = block_assemble(f)

U = BlockFunction(W)
block_solve(A, U.block_vector(), F)

plt.figure(1,(10,12))
plt.subplot('221')
plot(U[0][0])
plt.subplot('222')
plot(U[0][1])
plt.subplot('223')
plot(U[1][0])
plt.subplot('224')
plot(U[1][1])

# # Solving with standard Fenics
# u = TrialFunction(V)
# v = TestFunction(V)
# a = inner(grad(u), grad(v))*dx
# f = inner(v,Constant((1.0,1.0)))*dx
# A = assemble(a)
# F = assemble(f)
# bc = DirichletBC(V, g, mesh.boundaries, 1)
# bc.apply(A)
# bc.apply(F)

# U_ex = Function(V)

# solve(A, U_ex.vector(), F)

# plt.figure(3,(10,8))
# plt.subplot('121')
# plot(U_ex[0])
# plt.subplot('122')
# plot(U_ex[1])
# plt.show()

# # # error

# err = np.sqrt( assemble(dot(U_ex - U[0], U_ex - U[0])*dx))
# print("err", err)
