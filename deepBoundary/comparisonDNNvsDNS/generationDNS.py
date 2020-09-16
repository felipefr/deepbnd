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
import ioFenicsWrappers as iofe
import fenicsUtils as feut

import matplotlib.pyplot as plt


folder = "/Users/felipefr/EPFL/newDLPDES/DATA/deepBoundary/comparisonDNNvsDNS/seed2/"
#folder = "/home/felipefr/EPFL/newDLPDEs/DATA/deepBoundary/data2/"

radFile = folder + "RVE_POD_{0}.{1}"

opModel = 'periodic'

seed = 2
np.random.seed(seed)



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

maxOffset = 6
noutPerOffset = 5

for offset in range(1,maxOffset):
    Lx = Ly = 1.0
    ifPeriodic = False 
    NxL = NyL = 2
    NL = NxL*NyL
    x0L = y0L = offset*Lx/(NxL+2*offset)
    if(offset == 0 ):
        LxL = LyL = Lx 
    else:
        LxL = LyL = NxL*(x0L/offset)
    
    r0 = 0.2*LxL/NxL
    r1 = 0.4*LxL/NxL
    times = 1
    lcar = 0.1*LxL/NxL
    NpLx = int(Lx/lcar) + 1
    NpLxL = int(LxL/lcar) + 1
    Vfrac = 0.282743
    
    g = gmts.displacementGeneratorBoundary(x0L,y0L,LxL, LyL, NpLxL)
    
    
    print("nodes per side",  NpLx)
    print("nodes per internal side",  NpLxL )
    
    np.random.seed(seed)
    
    print('snapshot offset ', offset)
    ellipseDataCentral = geni.circularRegular2Regions(r0, r1, NxL, NyL, Lx, Ly, offset, ordered = True)[:NL,:]
    alphaFrac = np.sqrt((LxL*LyL)*Vfrac/(np.pi*np.sum(ellipseDataCentral[:,2]*ellipseDataCentral[:,2])))
    ellipseDataCentral[:,2] = alphaFrac*ellipseDataCentral[:,2]

    for nOut in range(noutPerOffset):
        ellipseDataOuter = geni.circularRegular2Regions(r0, r1, NxL, NyL, Lx, Ly, offset, ordered = True)[NL:,:]
        betaFrac = np.sqrt((1.0 - LxL*LyL)*Vfrac/(np.pi*np.sum(ellipseDataOuter[:,2]*ellipseDataOuter[:,2])))
        ellipseDataOuter[:,2] = betaFrac*ellipseDataOuter[:,2]
                     
        ellipseData = np.concatenate((ellipseDataCentral,ellipseDataOuter),axis = 0)
        
        np.savetxt(folder + 'ellipseData_offset{0}_{1}.txt'.format(offset,nOut), ellipseData)
        
        meshGMSH = gmsh.ellipseMesh2Domains(x0L, y0L, LxL, LyL, NL, ellipseData, Lx, Ly , lcar, ifPeriodic)
        meshGMSH.setTransfiniteBoundary(NpLx)
        meshGMSH.setTransfiniteInternalBoundary(NpLxL)
        
        meshGMSHred = gmsh.ellipseMesh2(ellipseDataCentral, x0L, y0L, LxL, LyL , lcar, ifPeriodic)
        meshGMSHred.setTransfiniteBoundary(NpLxL)
        
        mesh = feut.getMesh(meshGMSH, 'offset{0}_{1}'.format(offset,nOut), radFile)
        meshRed = feut.getMesh(meshGMSHred, 'reduced_offset{0}_{1}'.format(offset,nOut), radFile)

        BM = fmts.getBMrestriction(g, mesh)        

        sigma, sigmaEps = fmts.getSigma_SigmaEps(param,mesh,eps)
        
        # Solving with Multiphenics
        U = mpms.solveMultiscale(param, mesh, eps, op = opModel)
        T, a, B = feut.getAffineTransformationLocal(U[0],mesh,[0,1])
        Eps = - B.flatten()

        Utranslated = U[0] + T

        sigma_Total = fmts.homogenisation(U[0], mesh, sigma, [0,1,2,3], sigmaEps).flatten()
        sigma_MR = fmts.homogenisation(U[0], mesh, sigma, [0,1], sigmaEps).flatten()
                
        epsRed = -B + eps
        
        sigmaRed, sigmaEpsRed = fmts.getSigma_SigmaEps(param[0:2,:],meshRed,epsRed)
        Ured = mpms.solveMultiscale(param[0:2,:], meshRed, epsRed, op = 'BCdirich_lag', others = [Utranslated])
        sigma_red_MR = fmts.homogenisation(Ured[0], meshRed, sigmaRed, [0,1], sigmaEpsRed).flatten()
        
        print('sigma_MR:', sigma_MR)
        print('sigma_red_MR:', sigma_red_MR)
        print('sigma_MR - sigma_red_MR:', sigma_MR - sigma_red_MR)
        
        os.system("rm " + radFile.format('solution_red_offset{0}_{1}'.format(offset,nOut),'h5'))
        
        with HDF5File(MPI.comm_world, radFile.format('solution_red_offset{0}_{1}'.format(offset,nOut),'h5'), 'w') as f:
            f.write(Ured[0], 'basic')
                
        
        np.savetxt(folder + 'EpsList_offset{0}_{1}.txt'.format(offset,nOut), Eps)
        np.savetxt(folder + 'sigma_offset{0}_{1}.txt'.format(offset,nOut), sigma_MR)
        np.savetxt(folder + 'sigmaRed_offset{0}_{1}.txt'.format(offset,nOut), sigma_red_MR)
        np.savetxt(folder + 'sigmaTotal_offset{0}_{1}.txt'.format(offset,nOut), sigma_Total)
        
                    