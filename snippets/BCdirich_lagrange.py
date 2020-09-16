from __future__ import print_function
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from block import *
from block.iterative import *
from block.algebraic.petsc import *

mesh = UnitSquareMesh(10, 10)

TOL = 0.00001  
class FaceT(SubDomain):
  def inside(self, x, on_boundary):
    return near(x[0],1., TOL)

class FaceS(SubDomain):
  def inside(self, x, on_boundary):
    return near(x[0],0., TOL)

faceS = FaceS()
faceTd = FaceT()

bmesh = BoundaryMesh(mesh,'exterior') 
S_bmesh = SubMesh(bmesh,faceS) 

V = VectorFunctionSpace(mesh, "Lagrange", 1)
V0 = FunctionSpace(S_bmesh, "DG",0)
W = MixedFunctionSpace([V,V0])

bc1 = DirichletBC(W.sub(0), (0.0,0.0), faceT)
bcs = [bc1]

subdomains = MeshFunction("size_t", mesh, 1)
subdomains.set_all(0)
faceS.mark(subdomains, 1)
my_ds = Measure("ds", domain=mesh, subdomain_data=subdomains)

Exp = Expression('x[0] > 0.1 ? 1. : 0.0001')
project_x = Constant((1.,0.))

w  = Function(W)
(u,p) = split(w)

d = u.geometric_dimension()  
FF = Identity(d) + grad(u)                  
C = FF.T*FF                        

Ic = tr(C)
JJ  = det(FF)

# Elasticity parameters
mu, Kpar, Jm = 1., 10., 100.0

ux = dot(project_x,u)

psi = mu/2.*(Ic-2 - 2*ln(JJ))+Kpar/2.*(ln(JJ))**2
E = psi*dx + inner(p,(ux+0.1))*my_ds(1)+ Exp*inner(p,p)*dx

v  = TestFunction(W)
du = TrialFunction(W) 

J = derivative(derivative(E,w,v),w,du)
F = derivative(E,w,v)  

problem = NonlinearVariationalProblem(F, w, bcs, J=J)
solver  = NonlinearVariationalSolver(problem)
solver.parameters.nonlinear_solver = "snes"
solver.parameters.snes_solver.linear_solver = "umfpack"