import sys, os
from numpy import isclose
from dolfin import *
# import matplotlib.pyplot as plt
from multiphenics import *
sys.path.insert(0, '../../utils/')

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
import fenicsUtils as feut

from timeit import default_timer as timer


S = np.loadtxt('snapshots_MR_balanced_new.txt')
EpsFluc = np.loadtxt('EpsPer.txt')

folder = "./data1/"
radFile = folder + "RVE_POD_ref_{0}.{1}"

folder2 = "./dataConvergence/"
radFile2 = folder2 + "RVE_POD_ref_{0}.{1}"

offset = 2
Lx = Ly = 1.0
ifPeriodic = False 
NxL = NyL = 2
NL = NxL*NyL
x0L = y0L = offset*Lx/(NxL+2*offset)
LxL = LyL = offset*Lx/(NxL+2*offset)
r0 = 0.2*LxL/NxL
r1 = 0.4*LxL/NxL
times = 1
lcar = 0.05*LxL/NxL
NpLx = int(Lx/lcar) + 1
NpLxL = int(LxL/lcar) + 1

idEllipse = 0
facL = Lx/LxL

# S = facL*S

# Generation mesh complete
ellipseData = np.loadtxt(folder + 'ellipseData_' + str(idEllipse) + '.txt')
# ellipseData[:,2] = np.mean(ellipseData[:,2]) 

# meshGMSH1 = gmsh.ellipseMesh2(ellipseData[:NL], x0L, y0L, LxL, LyL, lcar, ifPeriodic)
# meshGMSH1.setTransfiniteBoundary(NpLxL)

meshGMSH1 = gmsh.ellipseMesh2Domains(x0L, y0L, LxL, LyL, NL, ellipseData, Lx, Ly , lcar, ifPeriodic)
meshGMSH1.setTransfiniteBoundary(NpLx)
meshGMSH1.setTransfiniteInternalBoundary(NpLxL)

# mesh1 = feut.getMesh(meshGMSH1, 'complete' , './mesh_{0}.{1}')
mesh1 = fela.EnrichedMesh('./mesh_complete.xml')

# Generation mesh reduced
meshGMSH2 = gmsh.ellipseMesh2(ellipseData[:NL], x0L, y0L, LxL, LyL, lcar, ifPeriodic)
meshGMSH2.setTransfiniteBoundary(NpLxL)

# mesh2 = feut.getMesh(meshGMSH2, 'reduced' , './mesh_{0}.{1}')
mesh2 = fela.EnrichedMesh('./mesh_reduced.xml')

# Solving mesh complete
eps = np.zeros((2,2))
eps[1,0] = 0.1
eps = 0.5*(eps + eps.T)

contrast = 10.0
E2 = 1.0
E1 = contrast*E2 # inclusions
nu1 = 0.3
nu2 = 0.3

mu1 = elut.eng2mu(nu1,E1)
lamb1 = elut.eng2lambPlane(nu1,E1)
mu2 = elut.eng2mu(nu2,E2)
lamb2 = elut.eng2lambPlane(nu2,E2)

param = np.array([[lamb1,mu1],[lamb2,mu2],[lamb1,mu1],[lamb2,mu2]])
param2 = np.array([[lamb1,mu1],[lamb2,mu2]])

U1 = mpms.solveMultiscale(param, mesh1, eps, op = 'MR', others = [])[0]

# Solving mesh reduced
gB = gmts.displacementGeneratorBoundary(0.0,0.0, Lx,Ly, NpLx)
gBL = gmts.displacementGeneratorBoundary(x0L,y0L,LxL,LyL, NpLxL)

T, a, B = feut.getAffineTransformationLocal(U1,mesh1,[0,1])
epsL = eps - B

U1L = fmts.PointExpression(gBL(mesh1,U1),gBL, 'python') + T
U1L_= gBL(mesh1,U1L)
BM = fmts.getBMrestriction(gBL, mesh1)
errorMR = BM.T @ U1L_
print('errorMR' , errorMR)

U2 = mpms.solveMultiscale(param2, mesh2, epsL, op = 'BCdirich_lag', others = [U1 + T])[0]

a2 = feut.Integral(U2, [mesh2.dx(0), mesh2.dx(1)], (2,))
B2 = feut.Integral(grad(U2), [mesh2.dx(0), mesh2.dx(1)], (2,2))
print(a2,B2)

# homogenisations
sigma1, sigmaEps1 = fmts.getSigma_SigmaEps(param,mesh1,eps)

sigma_hom_1 = fmts.homogenisation(U1,mesh1, sigma1, [0,1,2,3], sigmaEps1).flatten()
sigma_hom_1_L = fmts.homogenisation(U1,mesh1, sigma1, [0,1], sigmaEps1).flatten()

sigma2, sigmaEps2 = fmts.getSigma_SigmaEps(param2,mesh2,epsL)
sigma_hom_2 = fmts.homogenisation(U2,mesh2, sigma2, [0,1], sigmaEps2).flatten()

TT = feut.affineTransformationExpession(np.zeros(2), epsL, mesh2)
V = VectorFunctionSpace(mesh2,"CG", 1)
UU2 = interpolate(U2,V) + TT
sigma3, sigmaEps3 = fmts.getSigma_SigmaEps(param2,mesh2,np.zeros((2,2)))
sigma_hom_2_alt = fmts.homogenisation(UU2,mesh2, sigma3, [0,1], sigmaEps3).flatten()


print(sigma_hom_1_L, sigma_hom_2)
print(sigma_hom_1_L - sigma_hom_2)
print(sigma_hom_1_L - sigma_hom_2_alt)
V = VectorFunctionSpace(mesh2,"CG", 1)
U3 = interpolate(U1,V) + T

print('error L2 = ', np.sqrt(assemble(inner(U3-U2,U3-U2)*mesh2.dx)))
print('error L2 partial = ', np.sqrt(assemble(inner(U3-U2,U3-U2)*mesh2.ds)))

plt.figure(1,(10,10))
plt.subplot('221')
plot(U2[0])
plt.xlim(0.33333,0.666666)
plt.ylim(0.33333,0.666666)
plt.subplot('222')
plot(U2[1])
plt.xlim(0.33333,0.666666)
plt.ylim(0.33333,0.666666)
plt.subplot('223')
plot(U2[0])
plt.subplot('224')
plot(U2[1])

plt.show()