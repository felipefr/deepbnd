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
import multiphenicsMultiscale as mpms

# Put different materials V
# insert Boundary condition as points V
# Compute elasticity V
# Encapsulate elasticity V

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
              
Nside = 20
t = np.linspace(0.0,1.0,4*Nside)
uD_ = np.zeros(8*Nside) 
uD_[::2] = np.sin(15*np.pi*t)
uD_[1::2] = np.cos(8*np.pi*t)

# Solving with Multiphenics
U = mpms.solveMultiscale(param, mesh, eps, op = 'periodic')

plt.figure(1,(10,8))
plt.subplot('121')
plot(U[0][0])
plt.subplot('122')
plot(U[0][1])


# # Solving with standard Fenics
w = fmts.solveMultiscale(param, mesh, eps, op = 'periodic')

u_ex = w.split()[0]
p_ex = w.split()[1]

plt.figure(2,(10,8))
plt.subplot('121')
plot(u_ex[0])
plt.subplot('122')
plot(u_ex[1])
plt.show()

# # # # error
plt.figure(3,(10,8))
plt.subplot('121')
plot(u_ex[0] - U[0][0])
plt.subplot('122')
plot(u_ex[1] - U[0][1])
plt.show()

meanZero1 = assemble(u_ex[0]*dx)
meanZero2 = assemble(u_ex[1]*dx)
print(meanZero1, meanZero2)

err_u = np.sqrt( assemble(dot(u_ex - U[0], u_ex - U[0])*dx))
print("err_u", err_u)
