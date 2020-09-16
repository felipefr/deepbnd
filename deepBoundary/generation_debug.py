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
import ioFenicsWrappers as iofe
import fenicsUtils as feut

import matplotlib.pyplot as plt

folder = "./debugData/"
radFile = folder + "RVE_POD_{0}_.{1}"

np.random.seed(10)

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
lcar = 0.1*LxL/NxL
NpLx = int(Lx/lcar) + 1
NpLxL = int(LxL/lcar) + 1
Vfrac = 0.282743

print("nodes per side",  NpLx)
print("nodes per internal side",  NpLxL )

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

param = np.array([[lamb1, mu1], [lamb2,mu2],[lamb1, mu1], [lamb2,mu2]])

ns1 = 30
ns2 = 30

ns = ns1*ns2
S = np.zeros(((NpLxL-1)*8,ns))

g = gmts.displacementGeneratorBoundary(x0L,y0L,LxL, LyL, NpLxL)

seed = 4
np.random.seed(seed)
    
EpsFluc = np.zeros((ns,4))
Stress = np.zeros((ns,4))

BClabel = 'big_periodic'
BC = 'periodic'

for n1 in range(ns1):
    ellipseData1 = geni.circularRegular2Regions(r0, r1, NxL, NyL, Lx, Ly, offset, ordered = True)[NL:,:]
    betaFrac = np.sqrt((1.0 - LxL*LyL)*Vfrac/(np.pi*np.sum(ellipseData1[:,2]*ellipseData1[:,2])))
    ellipseData1[:,2] = betaFrac*ellipseData1[:,2]
             
    for n2 in range(ns2):
        n = n1*ns2 + n2
        print('snapshot ', n)
        
        ellipseData2 = geni.circularRegular2Regions(r0, r1, NxL, NyL, Lx, Ly, offset, ordered = True)[:NL,:]
        alphaFrac = np.sqrt((LxL*LyL)*Vfrac/(np.pi*np.sum(ellipseData2[:,2]*ellipseData2[:,2])))
        ellipseData2[:,2] = alphaFrac*ellipseData2[:,2]
        
        ellipseData = np.concatenate((ellipseData2,ellipseData1),axis = 0)
        
        np.savetxt(folder + 'ellipseData_' + BClabel + str(n) + '.txt', ellipseData)
        
        # Solving for complete
        
        meshGMSH = gmsh.ellipseMesh2Domains(x0L, y0L, LxL, LyL, NL, ellipseData, Lx, Ly , lcar, ifPeriodic)
        meshGMSH.setTransfiniteBoundary(NpLx)
        meshGMSH.setTransfiniteInternalBoundary(NpLxL)
        
        mesh = feut.getMesh(meshGMSH, str(n) + '_complete', radFile)
        
        sigma, sigmaEps = fmts.getSigma_SigmaEps(param,mesh,eps)
 
        U = mpms.solveMultiscale(param, mesh, eps, op = BC)[0]
        T, a, B = feut.getAffineTransformationLocal(U,mesh,[0,1])
        EpsFluc[n,:] = - B.flatten()

        S[:,n] = g(mesh,U + T).flatten()
        
        # sigma_all = fmts.homogenisation(U, mesh, sigma, [0,1,2,3], sigmaEps).flatten()
        Stress[n,:] = fmts.homogenisation(U, mesh, sigma, [0,1], sigmaEps).flatten()

        # Solving for reduced
        # meshGMSH2 = gmsh.ellipseMesh2(ellipseData[:NL], x0L, y0L, LxL, LyL, lcar, ifPeriodic)
        # meshGMSH2.setTransfiniteBoundary(NpLxL)
        
        # mesh2 = feut.getMesh(meshGMSH2, str(n) + '_reduced', radFile)
        
        # BM = fmts.getBMrestriction(g, mesh)        
        
        
        # UL = fmts.PointExpression(S[:,n],g, 'python') 
        # errorMR = BM.T @ S[:,n]
        # print('errorMR' , errorMR)

        # epsL = eps + EpsFluc[n,:].reshape((2,2))
        
        # U2 = mpms.solveMultiscale(param, mesh2, epsL, op = 'BCdirich_lag', others = [UL])[0]

        # sigmaL, sigmaEpsL = fmts.getSigma_SigmaEps(param[0:2,:],mesh2,epsL)
 
        # sigma_L_red= fmts.homogenisation(U2, mesh2, sigmaL, [0,1], sigmaEpsL).flatten()
        
        # print(sigma_all,sigma_L,sigma_L_red)

np.savetxt(folder + 'snapshots_{0}.txt'.format(BClabel), S)

np.savetxt(folder + 'Eps_{0}.txt'.format(BClabel), EpsFluc)

np.savetxt(folder + 'Stress_{0}.txt'.format(BClabel), Stress)
