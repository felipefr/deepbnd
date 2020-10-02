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


folder = "/Users/felipefr/EPFL/newDLPDES/DATA/deepBoundary/data3_justToCheck/"
#folder = "/home/felipefr/EPFL/newDLPDEs/DATA/deepBoundary/data2/"

radFile = folder + "RVE_POD_{0}.{1}"

opModel = 'periodic'

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

Neps = 1
epsList = [np.zeros((2,2)),np.zeros((2,2)),np.zeros((2,2))]
epsList[0][0,0] = 1.0
epsList[1][1,1] = 1.0
epsList[2][1,0] = 0.5
epsList[2][0,1] = 0.5

param = np.array([[lamb1, mu1], [lamb2,mu2],[lamb1, mu1], [lamb2,mu2]])

ns1 = 1
ns2 = 10

ns = ns1*ns2

g = gmts.displacementGeneratorBoundary(x0L,y0L,LxL, LyL, NpLxL)

seed = 3
np.random.seed(seed)
    
Eps = np.zeros((ns,Neps,4))
NodesBoundary = np.zeros((4*(NpLxL-1),ns))

sigmaList = np.zeros((ns,Neps,4))

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
        
        meshGMSH = gmsh.ellipseMesh2Domains(x0L, y0L, LxL, LyL, NL, ellipseData, Lx, Ly , lcar, ifPeriodic)
        meshGMSH.setTransfiniteBoundary(NpLx)
        meshGMSH.setTransfiniteInternalBoundary(NpLxL)
        
        meshGMSHred = gmsh.ellipseMesh2(ellipseData2, x0L, y0L, LxL, LyL , lcar, ifPeriodic)
        meshGMSHred.setTransfiniteBoundary(NpLxL)
        
        mesh = feut.getMesh(meshGMSH, str(n), radFile)
        meshRed = feut.getMesh(meshGMSHred, 'reduced_' + str(n), radFile)

        BM = fmts.getBMrestriction(g, mesh)        

        NodesBoundary[:,n] = g.identifyNodes(meshRed)
        np.savetxt(folder + 'NodesBoundary.txt', NodesBoundary)

        for n3 in range(Neps):        
            eps = epsList[n3]
            sigma, sigmaEps = fmts.getSigma_SigmaEps(param,mesh,eps)
            
            # Solving with Multiphenics
            U = mpms.solveMultiscale(param, mesh, eps, op = opModel)
            T, a, B = feut.getAffineTransformationLocal(U[0],mesh,[0,1])
            Eps[n,n3,:] = - B.flatten()
    
            Utranslated = U[0] + T
    
            sigma_MR = fmts.homogenisation(U[0], mesh, sigma, [0,1], sigmaEps).flatten()
                    
            epsRed = -B + eps
            
            sigmaRed, sigmaEpsRed = fmts.getSigma_SigmaEps(param[0:2,:],meshRed,epsRed)
            Ured = mpms.solveMultiscale(param[0:2,:], meshRed, epsRed, op = 'BCdirich_lag', others = [Utranslated])
            sigma_red_MR = fmts.homogenisation(Ured[0], meshRed, sigmaRed, [0,1], sigmaEpsRed).flatten()
            
            print('sigma_MR:', sigma_MR)
            print('sigma_red_MR:', sigma_red_MR)
            print('sigma_MR - sigma_red_MR:', sigma_MR - sigma_red_MR)
            
            os.system("rm " + radFile.format('solution_red','h5'))
            
            with HDF5File(MPI.comm_world, radFile.format('solution_red_' + str(n) + "_" + str(n3),'h5'), 'w') as f:
                f.write(Ured[0], 'basic')
                    
            iofe.postProcessing_complete(U[0], folder + 'sol_mesh_{0}_{1}.xdmf'.format(n,n3), ['u','lame','vonMises'], param)
                    
                       
            sigmaList[n,n3,:] = sigma_red_MR
            
            np.savetxt(folder + 'EpsList_' + str(n3) + '.txt', Eps[:,n3,:])
            np.savetxt(folder + 'sigmaList' + str(n3) + '.txt', sigmaList[:,n3,:])
        
                    
        
# with HDF5File(MPI.comm_world, radFile.format('solution_red','h5'), 'r') as f:
#     f.read(Ured[0], 'basic')
    
# iofe.postProcessing_complete(U[0], 'sol.xdmf', ['u','lame','vonMises'], param)
    
# plt.show()
# np.savetxt('snapshots_MR_balanced_new.txt', SMR)
# np.savetxt('snapshots_periodic_balanced_new.txt', Sper)

# np.savetxt(folder + 'EpsPer.txt', Eps)
# np.savetxt(folder + 'NodesBoundary.txt', NodesBoundary)
# np.savetxt(folder + 'sigmaList.txt', sigmaList)

# V = VectorFunctionSpace(meshRed,"CG", 1)
# nodes = g.identifyNodes(meshRed)
# u = Ured[0]

# v2d = vertex_to_dof_map(V) # used to take a vector in its dofs and transform to dofs according to node numbering in mesh

# for i in range(80):
#     print(u.vector().get_local()[v2d[2*nodes[i]:2*(nodes[i]+1)]],S[2*i:2*(i+1),0])
