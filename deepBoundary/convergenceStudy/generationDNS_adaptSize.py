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
import meshUtils as meut
import generationInclusions as geni
import myCoeffClass as coef
import fenicsMultiscale as fmts
import elasticity_utils as elut
import fenicsWrapperElasticity as fela
import multiphenicsMultiscale as mpms
import ioFenicsWrappers as iofe
import fenicsUtils as feut

import matplotlib.pyplot as plt
import copy

# ellipseData[permBox[4:],2] = outerRadius
# ellipseData = ellipseData[permTotal]

# radius = ellipseData[:,2].reshape((Nx,Ny))
# alpha = 2*H*np.sqrt(Vfrac/(np.pi*np.sum(radius[maxOffset : maxOffset + NxL, maxOffset : maxOffset + NxL]**2)))
# radius[maxOffset : maxOffset + NxL, maxOffset : maxOffset + NxL] *= alpha

def enforceVfracPerOffset(radius, NxL, maxOffset, H, Vfrac): # radius should be ordened interior to exterior, 
    for i in range(maxOffset+1):
        ni =  (NxL + 2*(i-1))**2 
        nout = (NxL + 2*i)**2
        alphaFrac = H*np.sqrt((nout-ni)*Vfrac/(np.pi*np.sum(radius[ni:nout]**2)))
        radius[ni:nout] *= alphaFrac
        
    return radius


folder = ["/Users", "/home"][0] + "/felipefr/EPFL/newDLPDEs/DATA/deepBoundary/convergenceStudy/adaptSize/"

radFile = folder + "RVE_POD_{0}.{1}"

opModel = 'Lin'
createMesh = True

contrast = 10.0
E2 = 1.0
E1 = contrast*E2 # inclusions
nu1 = 0.3
nu2 = 0.3

mu1 = elut.eng2mu(nu1,E1)
lamb1 = elut.eng2lambPlane(nu1,E1)
mu2 = elut.eng2mu(nu2,E2)
lamb2 = elut.eng2lambPlane(nu2,E2)

# tension test
# eps = np.zeros((2,2))
# eps[0,0] = 1.0 # after rescaled to have 1.0

# shear test
eps = np.zeros((2,2))
eps[0,1] = 0.5
eps[1,0] = 0.5

param = np.array([[lamb1, mu1], [lamb2,mu2],[lamb1, mu1], [lamb2,mu2]])

Offset0 = 0
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
lcar = 0.1*LxL/NxL
lcar2 = 2.*lcar
NpLx = int(Lx/lcar2) + 1
NpLxL = int(LxL/lcar) + 1
Vfrac = 0.282743
H = Lx/Nx
rm = H*np.sqrt(Vfrac/np.pi)

# maxOffset = 7 # we use meshes build with 8 offset, but let us simulate until 7 (important for peridioc, since the BC is important)

for offset in range(Offset0,maxOffset + 1):
    for seed in range(0,20):

        Nt = (NxL + 2*offset)**2
        Lxt = Lyt =  H*np.sqrt(Nt)
        NpLxt = int(Lxt/lcar) + 1
         
        x0 = x0L - offset*H; y0 = y0L - offset*H
               
        mesh = meut.EnrichedMesh(folder + "mesh_offset{0}_{1}.{2}".format(offset,seed,'xdmf'))
         
        sigma, sigmaEps = fmts.getSigma_SigmaEps(param,mesh,eps, op = 'cpp')
        
        bdr = 2 if offset == 0 else 4
        
        for opModel in ['periodic','MR', 'Lin']:
            # Solving with Multiphenics

            U = mpms.solveMultiscale(param, mesh, eps, op = opModel, others = {'polyorder' : 2, 'bdr' : bdr, 'per': [x0, x0 + Lxt, y0, y0 + Lyt]})
        
            sigma_T = fmts.homogenisation(U[0], mesh, sigma, [0,1,2,3], sigmaEps).flatten()
            sigma_L = fmts.homogenisation(U[0], mesh, sigma, [0,1], sigmaEps).flatten()              
            
            np.savetxt(folder + 'shear/sigmaL_{0}_offset{1}_{2}.txt'.format(opModel,offset,seed), sigma_L)
            np.savetxt(folder + 'shear/sigmaT_{0}_offset{1}_{2}.txt'.format(opModel,offset,seed), sigma_T)
        

# for seed in range(0,20):
#     print(seed)
#     np.random.seed(seed)
#     ellipseData, PermTotal, PermBox = geni.circularRegular2Regions(r0, r1, NxL, NyL, Lx, Ly, maxOffset, ordered = False)
#     ellipseData = ellipseData[PermTotal]
    
#     ellipseData[:,2] = enforceVfracPerOffset(ellipseData[:,2], NxL, maxOffset, H, Vfrac)
#     ellipseData[NxL**2:,2] = rm
#     np.savetxt(folder + 'ellipseData_{0}.txt'.format(seed), ellipseData)
    
#     for offset in range(Offset0,maxOffset + 1):
                     
#         Nt = (NxL + 2*offset)**2
#         Lxt = Lyt =  H*np.sqrt(Nt)
#         NpLxt = int(Lxt/lcar2) + 1
         
#         x0 = x0L - offset*H; y0 = y0L - offset*H
                
#         if(offset == 0):
#             meshGMSH = meut.ellipseMesh2(ellipseData[:Nt], x0 = x0L, y0 = x0L, Lx = LxL , Ly = LxL , lcar = lcar)
#             meshGMSH.setTransfiniteBoundary(NpLxL)
        
#         else:
#             meshGMSH = meut.ellipseMesh2Domains(x0L, y0L, LxL, LyL, NL, ellipseData[:Nt], Lxt, Lyt, 4*[lcar] + (Nt-4)*[lcar2], x0 = x0, y0 = y0)
#             meshGMSH.setTransfiniteBoundary(NpLxt)
#             meshGMSH.setTransfiniteInternalBoundary(NpLxL)   

#         meshGMSH.setNameMesh(folder + "mesh_offset{0}_{1}.{2}".format(offset,seed,'xdmf'))
#         mesh = meshGMSH.getEnrichedMesh() 


# os.system("rm " + radFile.format('solRed_{0}_offset{1}_{2}'.format(opModel,offset,seed),'h5'))

# with HDF5File(MPI.comm_world, radFile.format('solRed_{0}_offset{1}_{2}'.format(opModel,offset,seed),'h5'), 'w') as f:
#     f.write(U[0], 'basic')
        
# iofe.postProcessing_complete(U[0], folder + 'sol_mesh_{0}_offset{1}_{2}.xdmf'.format(opModel,offset,seed), ['u','lame','vonMises'], param)       

