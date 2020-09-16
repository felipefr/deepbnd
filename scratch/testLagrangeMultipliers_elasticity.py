import sys, os
from numpy import isclose
from dolfin import *
# import matplotlib.pyplot as plt
from multiphenics import *
sys.path.insert(0, '../utils/')

import fenicsWrapperElasticity as fela
import matplotlib.pyplot as plt
import numpy as np
import generatorMultiscale as gmts
import wrapperPygmsh as gmsh
import generationInclusions as geni
import myCoeffClass as coef
import fenicsMultiscale as fmts
import elasticity_utils as elut
import fenicsWrapperElasticity as fela


# Put different materials V
# insert Boundary condition as points V
# Compute elasticity V
# Encapsulate elasticity

# Mesh creation
folder = "./data/"
radFile = folder + "RVE_{0}_{1}.{2}"

np.random.seed(10)

offset = 0
Lx = Ly = 1.0
ifPeriodic = False 
NxL = NyL = 2
x0L = y0L = offset*Lx/(1+2*offset)
LxL = LyL = Lx/(1+2*offset)
r0 = 0.2*LxL/NxL
r1 = 0.4*LxL/NxL

ellipseData = geni.circularRegular2Regions(r0, r1, NxL, NyL, Lx, Ly, offset, ordered = True)

times = 1

lcar = 0.1*Lx/(NxL*times)

meshGMSH = gmsh.ellipseMeshRepetition(times, ellipseData, Lx, Ly , lcar, ifPeriodic)
meshGMSH.setTransfiniteBoundary(int(Lx/lcar) + 1)
print("nodes per side",  int(Lx/lcar) + 1)

meshGeoFile = radFile.format(times,'','geo')
meshXmlFile = radFile.format(times,'','xml')
meshMshFile = radFile.format(times,'','msh')

meshGMSH.write(meshGeoFile,'geo')
os.system('gmsh -2 -algo del2d -format msh2 ' + meshGeoFile)

os.system('dolfin-convert {0} {1}'.format(meshMshFile, meshXmlFile))

mesh = fela.EnrichedMesh(meshXmlFile)

# end mesh creation

materials = mesh.subdomains.array().astype('int32')
materials = materials - np.min(materials)

contrast = 10.0
E2 = 1.0
E1 = contrast*E2 # inclusions
nu1 = 0.3
nu2 = 0.3

mu1 = elut.eng2mu(nu1,E1)
lamb1 = elut.eng2lambPlane(nu1,E1)
mu2 = elut.eng2mu(nu2,E2)
lamb2 = elut.eng2lambPlane(nu2,E2)

eps = np.zeros((2,2))
eps[1,0] = 0.5
eps = 0.5*(eps + eps.T)

param = np.array([[lamb1, mu1], [lamb2,mu2]])

sigma, sigmaEps = fmts.getSigma_SigmaEps(param,mesh,eps)    
              
class OnBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

onBoundary = OnBoundary()
boundary_restriction = MeshRestriction(mesh, onBoundary)

VE = VectorElement("Lagrange", mesh.ufl_cell(), 1)
RE = VectorElement("Real", mesh.ufl_cell(), 0)

V = FunctionSpace(mesh, VE )
R = FunctionSpace(mesh, RE )

Nside = 20
t = np.linspace(0.0,1.0,4*Nside)
uD_ = np.zeros(8*Nside) 
uD_[::2] = np.sin(15*np.pi*t)
uD_[1::2] = np.cos(8*np.pi*t)
gen = gmts.displacementGeneratorBoundary(0.0,0.0,1.0,1.0, Nside + 1)
g = fmts.PointExpression(uD_, gen)

# Solving with Multiphenics
W = BlockFunctionSpace([V, V, R], restrict=[None, boundary_restriction, None])

ulp = BlockTrialFunction(W)
(u, l, p) = block_split(ulp)
vmq = BlockTestFunction(W)
(v, m, q) = block_split(vmq)

a = []
a.append([ inner(sigma(u),fela.epsilon(v))*dx, inner(l,v)*ds, inner(p,v)*dx ])
a.append([inner(u,m)*ds                    , 0, 0     ])
a.append([inner(u,q)*dx                    , 0, 0     ])

f =  [ -inner(sigmaEps, fela.epsilon(v))*dx , inner(g,m)*ds , 0]

A = block_assemble(a)
F = block_assemble(f)

U = BlockFunction(W)
block_solve(A, U.block_vector(), F)

plt.figure(1,(10,8))
plt.subplot('121')
plot(U[0][0])
plt.subplot('122')
plot(U[0][1])

# # Solving with standard Fenics

W = FunctionSpace(mesh, MixedElement([VE, RE]))   
u1,u2,p1,p2 = TrialFunction(W)
v1,v2,q1,q2 = TestFunction(W)
    
p = as_vector((p1,p2))
q = as_vector((q1,q2))
u = as_vector((u1,u2))
v = as_vector((v1,v2))

a = inner(sigma(u),fela.epsilon(v))*dx +  inner(p,v)*dx + inner(q,u)*dx
f = -inner(sigmaEps, fela.epsilon(v))*dx
A = assemble(a)
F = assemble(f)
bc = DirichletBC(W.sub(0), g, mesh.boundaries, 2)
bc.apply(A)
bc.apply(F)

w = Function(W)

solve(A, w.vector(), F)

U_ex = w.split()[0]
P_ex = w.split()[1]


plt.figure(2,(10,8))
plt.subplot('121')
plot(U_ex[0])
plt.subplot('122')
plot(U_ex[1])
plt.show()

# # # # error
plt.figure(3,(10,8))
plt.subplot('121')
plot(U_ex[0] - U[0][0])
plt.subplot('122')
plot(U_ex[1] - U[0][1])
plt.show()

meanZero1 = assemble(U_ex[0]*dx)
meanZero2 = assemble(U_ex[1]*dx)
print(meanZero1, meanZero2)

errU = np.sqrt( assemble(dot(U_ex - U[0], U_ex - U[0])*dx))
errP = np.sqrt( assemble(dot(P_ex - U[2], P_ex - U[2])*dx))

print("errU", errU, "errP", errP)
