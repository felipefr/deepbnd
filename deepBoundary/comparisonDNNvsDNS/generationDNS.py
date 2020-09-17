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


folder = "/Users/felipefr/EPFL/newDLPDES/DATA/deepBoundary/comparisonDNNvsDNS/"
#folder = "/home/felipefr/EPFL/newDLPDEs/DATA/deepBoundary/data2/"

radFile = folder + "RVE_POD_{0}.{1}"

opModel = 'periodic'
createMesh = True

seed = 1

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

param = np.array([[lamb1, mu1], [lamb2,mu2],[lamb1, mu1], [lamb2,mu2]])

maxOffset = 8

Lx = Ly = 1.0
ifPeriodic = False 
NxL = NyL = 2
Nx = (NxL+2*maxOffset)
Ny = (NyL+2*maxOffset)
NL = NxL*NyL
x0L = y0L = maxOffset*Lx/Nx
LxL = LyL = NxL*(x0L/maxOffset)
r0 = 0.2*LxL/NxL
r1 = 0.4*LxL/NxL
lcar = 0.05*LxL/NxL
NpLx = int(Lx/lcar) + 1
NpLxL = int(LxL/lcar) + 1
Vfrac = 0.282743

H = Lx/Nx

g = gmts.displacementGeneratorBoundary(x0L,y0L,LxL, LyL, NpLxL)


np.random.seed(seed)
ellipseData = geni.circularRegular2Regions(r0, r1, NxL, NyL, Lx, Ly, maxOffset, ordered = True)

# enforcing volume frac
for i in range(maxOffset):
    ni =  (Nx - 2*(i+1))*(Ny - 2*(i+1))
    nout = (Nx - 2*i)*(Ny - 2*i)
    
    alphaFrac = np.sqrt(((nout-ni)*H**2 - LxL*LyL)*Vfrac/(np.pi*np.sum(ellipseData[ni:nout,2]*ellipseData[ni:nout,2])))
    ellipseData[ni:nout,2] = alphaFrac*ellipseData[ni:nout,2]

for offset in range(maxOffset+1):
                 
    # np.savetxt(folder + 'ellipseData_offset{0}_{1}.txt'.format(offset,nOut), ellipseData)
    Nt = (NxL + 2*offset)**2
    Lxt = Lyt =  H*np.sqrt(Nt)
    NpLxt = int(Lxt/lcar) + 1
        
    meshGMSHred = gmsh.ellipseMesh2(ellipseData[:NL], x0L, y0L, LxL, LyL , lcar, ifPeriodic)
    meshGMSHred.setTransfiniteBoundary(NpLxL)
    meshRed = feut.getMesh(meshGMSHred, 'reduced_offset{0}_{1}'.format(offset,seed), radFile, createMesh)
    
    x0 = x0L - offset*H; y0 = y0L - offset*H
    
    if(offset == 0):
        meshGMSH = meshGMSHred
        mesh = meshRed
    else:
        meshGMSH = gmsh.ellipseMesh2Domains(x0L, y0L, LxL, LyL, NL, ellipseData[:Nt], Lxt, Lyt, lcar, x0 = x0, y0 = y0)
        meshGMSH.setTransfiniteBoundary(NpLxt)
        meshGMSH.setTransfiniteInternalBoundary(NpLxL)
        mesh = feut.getMesh(meshGMSH, 'offset{0}_{1}'.format(offset,seed), radFile, createMesh)


    BM = fmts.getBMrestriction(g, mesh)        

    sigma, sigmaEps = fmts.getSigma_SigmaEps(param,mesh,eps)
    
    # Solving with Multiphenics
    U = mpms.solveMultiscale(param, mesh, eps, op = opModel, others = [x0, x0 + Lxt, y0, y0 + Lyt])
    T, a, B = feut.getAffineTransformationLocal(U[0],mesh,[0,1])
    Eps = - B.flatten()

    Utranslated = U[0] + T

    sigma_T = fmts.homogenisation(U[0], mesh, sigma, [0,1,2,3], sigmaEps).flatten()
    sigma_L = fmts.homogenisation(U[0], mesh, sigma, [0,1], sigmaEps).flatten()
            
    epsRed = -B + eps
    
    sigmaRed, sigmaEpsRed = fmts.getSigma_SigmaEps(param[0:2,:],meshRed,epsRed)
    Ured = mpms.solveMultiscale(param[0:2,:], meshRed, epsRed, op = 'BCdirich_lag', others = [Utranslated])
    # sigma_red_MR = fmts.homogenisation(Ured[0], meshRed, sigmaRed, [0,1], sigmaEpsRed).flatten()
    
    print('sigma_L:', sigma_L)
    
    os.system("rm " + radFile.format('solRed_{0}_offset{1}_{2}'.format(opModel,offset,seed),'h5'))
    
    with HDF5File(MPI.comm_world, radFile.format('solRed_{0}_offset{1}_{2}'.format(opModel,offset,seed),'h5'), 'w') as f:
        f.write(Ured[0], 'basic')
            
    iofe.postProcessing_complete(U[0], folder + 'sol_mesh_{0}_offset{1}_{2}.xdmf'.format(opModel,offset,seed), ['u','lame','vonMises'], param)
    
    np.savetxt(folder + 'EpsList_{0}_offset{1}_{2}.txt'.format(opModel,offset,seed), Eps)
    np.savetxt(folder + 'sigmaL_{0}_offset{1}_{2}.txt'.format(opModel,offset,seed), sigma_L)
    np.savetxt(folder + 'sigmaT_{0}_offset{1}_{2}.txt'.format(opModel,offset,seed), sigma_T)
        
                    