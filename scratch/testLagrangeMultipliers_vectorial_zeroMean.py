import sys
from numpy import isclose
from dolfin import *
# import matplotlib.pyplot as plt
from multiphenics import *
sys.path.insert(0, '../utils/')

import fenicsWrapperElasticity as fela
import matplotlib.pyplot as plt
import numpy as np

# change mesh

mesh = fela.EnrichedMesh("data/circle.xml")

class OnBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

onBoundary = OnBoundary()
boundary_restriction = MeshRestriction(mesh, onBoundary)

VE = VectorElement("Lagrange", mesh.ufl_cell(), 1)
RE = VectorElement("Real", mesh.ufl_cell(), 0)


V = FunctionSpace(mesh, VE )
R = FunctionSpace(mesh, RE )

g = Expression(("sin(3*x[0] + 1)*sin(3*x[1] + 1)","cos(x[0] + 1)*cos(x[1] + 1)"), element=V.ufl_element())

# Solving with Multiphenics
W = BlockFunctionSpace([V, V, R], restrict=[None, boundary_restriction, None])

ulp = BlockTrialFunction(W)
(u, l, p) = block_split(ulp)
vmq = BlockTestFunction(W)
(v, m, q) = block_split(vmq)

a = []
a.append([inner(grad(u), grad(v))*dx, inner(l,v)*ds, inner(p,v)*dx ])
a.append([inner(u,m)*ds                    , 0, 0     ])
a.append([inner(u,q)*dx                    , 0, 0     ])

f =  [ inner(v,Constant((1.0,1.0)))*dx , inner(g,m)*ds , 0]

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
W = FunctionSpace(mesh, MixedElement([VE, RE]))   
u1,u2,p1,p2 = TrialFunction(W)
v1,v2,q1,q2 = TestFunction(W)
    
p = as_vector((p1,p2))
q = as_vector((q1,q2))
u = as_vector((u1,u2))
v = as_vector((v1,v2))

a = inner(grad(u), grad(v))*dx +  inner(p,v)*dx + inner(q,u)*dx
f = inner(v,Constant((1.0,1.0)))*dx
A = assemble(a)
F = assemble(f)
bc = DirichletBC(W.sub(0), g, mesh.boundaries, 1)
bc.apply(A)
bc.apply(F)

w = Function(W)

solve(A, w.vector(), F)

U_ex = w.split()[0]
P_ex = w.split()[1]


plt.figure(3,(10,8))
plt.subplot('121')
plot(U_ex[0])
plt.subplot('122')
plot(U_ex[1])
plt.show()

# # # error
meanZero1 = assemble(U_ex[0]*dx)
meanZero2 = assemble(U_ex[1]*dx)
print(meanZero1, meanZero2)

errU = np.sqrt( assemble(dot(U_ex - U[0], U_ex - U[0])*dx))
errP = np.sqrt( assemble(dot(P_ex - U[2], P_ex - U[2])*dx))

print("errU", errU, "errP", errP)
