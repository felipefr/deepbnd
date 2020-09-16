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


folder = "./data/"
radFile = folder + "RVE_POD_{0}_.{1}"

np.random.seed(10)

offset = 0
Lx = Ly = 1.0
ifPeriodic = False 
NxL = NyL = 2
x0L = y0L = offset*Lx/(1+2*offset)
LxL = LyL = Lx/(1+2*offset)
r0 = 0.2*LxL/NxL
r1 = 0.4*LxL/NxL
times = 1
lcar = 0.1*Lx/(NxL*times)
print("nodes per side",  int(Lx/lcar) + 1)

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
eps[0,0] = 0.1
eps = 0.5*(eps + eps.T)

param = np.array([[lamb1, mu1], [lamb2,mu2]])

ns = 400
S = np.zeros((160,ns))
g = gmts.displacementGeneratorBoundary(0.0,0.0,1.0,1.0, 21)
    
for n in range(ns):
    ellipseData = geni.circularRegular2Regions(r0, r1, NxL, NyL, Lx, Ly, offset, ordered = True)
    meshGMSH = gmsh.ellipseMeshRepetition(times, ellipseData, Lx, Ly , lcar, ifPeriodic)
    meshGMSH.setTransfiniteBoundary(int(Lx/lcar) + 1)
    
    meshGeoFile = radFile.format(n,'geo')
    meshXmlFile = radFile.format(n,'xml')
    meshMshFile = radFile.format(n,'msh')
    
    meshGMSH.write(meshGeoFile,'geo')
    os.system('gmsh -2 -algo del2d -format msh2 ' + meshGeoFile)
    
    os.system('dolfin-convert {0} {1}'.format(meshMshFile, meshXmlFile))

    mesh = fela.EnrichedMesh(meshXmlFile)
              
    # Solving with Multiphenics
    U = mpms.solveMultiscale(param, mesh, eps, op = 'MR')

    S[:,n] = g(mesh,U[0]).flatten()
    

np.savetxt('snapshots.txt', S)