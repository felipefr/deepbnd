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

folder = "./data1/"
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

ns1 = 1
ns2 = 1

ns = ns1*ns2
Sper = np.zeros(((NpLxL-1)*8,ns))
SMR = np.zeros(((NpLxL-1)*8,ns))

g = gmts.displacementGeneratorBoundary(x0L,y0L,LxL, LyL, NpLxL)


seed = 1
np.random.seed(seed)
    
EpsPer = np.zeros((ns,4))
EpsMR = np.zeros((ns,4))

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
        
        np.savetxt(folder + 'ellipseData_' + str(n) + '.txt', ellipseData)
        
        meshGMSH = gmsh.ellipseMeshRepetition(times, ellipseData, Lx, Ly , lcar, ifPeriodic)
        meshGMSH = gmsh.ellipseMesh2Domains(x0L, y0L, LxL, LyL, NL, ellipseData, Lx, Ly , lcar, ifPeriodic)
        meshGMSH.setTransfiniteBoundary(NpLx)
        meshGMSH.setTransfiniteInternalBoundary(NpLxL)
        
        meshGeoFile = radFile.format(n,'geo')
        meshXmlFile = radFile.format(n,'xml')
        meshMshFile = radFile.format(n,'msh')
        
        meshGMSH.write(meshGeoFile,'geo')
        os.system('gmsh -2 -algo del2d -format msh2 ' + meshGeoFile)
        
        os.system('dolfin-convert {0} {1}'.format(meshMshFile, meshXmlFile))

        mesh = fela.EnrichedMesh(meshXmlFile)
        
        BM = fmts.getBMrestriction(g, mesh)        
        
        sigma, sigmaEps = fmts.getSigma_SigmaEps(param,mesh,eps)


        # Solving with Multiphenics
        UMR = mpms.solveMultiscale(param, mesh, eps, op = 'MR')
        T, a, B = feut.getAffineTransformationLocal(UMR[0],mesh,[0,1])
        EpsMR[n,:] = - B.flatten()

        SMR[:,n] = g(mesh,UMR[0] + T).flatten()
        
        print('error MR . :', BM.T @ SMR[:,n])
        sigma_MR = fmts.homogenisation(UMR[0], mesh, sigma, [0,1], sigmaEps).flatten()
        print('sigma MR:', sigma_MR)
        
        Uper = mpms.solveMultiscale(param, mesh, eps, op = 'periodic')        
        T, a, B = feut.getAffineTransformationLocal(Uper[0],mesh,[0,1])
        EpsPer[n,:] = - B.flatten()
        
        Sper[:,n] = g(mesh,Uper[0] + T).flatten()
        print('error MR:', BM.T @ Sper[:,n])
        sigma_per = fmts.homogenisation(Uper[0], mesh, sigma, [0,1], sigmaEps).flatten()
        print('sigma per:', sigma_per)
        
        
        
    # iofe.postProcessing_complete(U[0], 'sol.xdmf', ['u','lame','vonMises'], param)
    
# plt.show()
# np.savetxt('snapshots_MR_balanced_new.txt', SMR)
# np.savetxt('snapshots_periodic_balanced_new.txt', Sper)

# np.savetxt('EpsMR.txt', EpsMR)
# np.savetxt('EpsPer.txt', EpsPer)
