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
import myHDF5 as myhd
import matplotlib.pyplot as plt
import copy

def enforceVfracPerOffset(radius, NxL, maxOffset, H, Vfrac): # radius should be ordened interior to exterior, 
    for i in range(maxOffset+1):
        ni =  (NxL + 2*(i-1))**2 
        nout = (NxL + 2*i)**2
        alphaFrac = H*np.sqrt((nout-ni)*Vfrac/(np.pi*np.sum(radius[ni:nout]**2)))
        radius[ni:nout] *= alphaFrac
        
    return radius

folder = ["/Users", "/home"][0] + "/felipefr/EPFL/newDLPDEs/DATA/deepBoundary/generateDataset/3x3/"

radFile = folder + "RVE_POD_{0}.{1}"

opModel = 'periodic'
opLoad = 'axial'
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
param = np.array([[lamb1, mu1], [lamb2,mu2],[lamb1, mu1], [lamb2,mu2]])

# tension test
epsTension = np.zeros((2,2))
epsTension[0,0] = 1.0 # after rescaled to have 1.0

# shear test
epsShear = np.zeros((2,2))
epsShear[0,1] = 0.5
epsShear[1,0] = 0.5

eps = epsTension if opLoad == 'axial' else epsShear

maxOffset = 6

H = 1.0 # size of each square
NxL = NyL = 2
NL = NxL*NyL
x0L = y0L = -H 
LxL = LyL = 2*H
lcar = 0.1*H
Nx = (NxL+2*maxOffset)
Ny = (NyL+2*maxOffset)
Lxt = Nx*H
Lyt = Ny*H
NpLxt = int(Lxt/lcar) + 1
NpLxL = int(LxL/lcar) + 1
x0 = -Lxt/2.0
y0 = -Lyt/2.0
r0 = 0.2*H
r1 = 0.4*H
Vfrac = 0.282743
rm = H*np.sqrt(Vfrac/np.pi)

# meshRef = meut.degeneratedBoundaryRectangleMesh(x0 = x0L, y0 = y0L, Lx = LxL , Ly = LyL , Nb = 50)
# meshRef.generate()
# meshRef.write('boundaryMesh.xdmf', 'fenics')
Mref = meut.EnrichedMesh('boundaryMesh.xdmf')
Vref = VectorFunctionSpace(Mref,"CG", 1)
usol = Function(Vref)

ns = 1100

# Radius Generation
seed = 7
np.random.seed(seed)

os.system('rm ' + folder +  'ellipseData_{0}.h5'.format(seed))
X, f = myhd.zeros_openFile(filename = folder +  'ellipseData_{0}.h5'.format(seed),  shape = (ns,Ny*Nx,5), label = 'ellipseData', mode = 'w-')

for i in range(ns):
    ellipseData, PermTotal, PermBox = geni.circularRegular2Regions(r0, r1, NxL, NyL, Lxt, Lyt, maxOffset, ordered = False, x0 = x0, y0 = y0)
    ellipseData = ellipseData[PermTotal]
    
    ellipseData[:,2] = enforceVfracPerOffset(ellipseData[:,2], NxL, maxOffset, H, Vfrac)
    ellipseData[(NxL + 2)**2:,2] = rm # enforcing same radius after the first after the 
    
    X[i,:,:] = ellipseData

f.close()


os.system('rm ' + folder +  '{0}/snapshots_{1}.h5'.format(opLoad,seed))
snapshots, fsnaps = myhd.zeros_openFile(filename = folder +  '{0}/snapshots_{1}.h5'.format(opLoad, seed),  
                                        shape = [(ns,Vref.dim()),(ns,3),(ns,2),(ns,2,2)], label = ['solutions','sigma','a','B'], mode = 'w-')

snap_solutions, snap_sigmas, snap_a, snap_B = snapshots
     
ellipseData, fellipse = myhd.loadhd5_openFile(folder +  'ellipseData_{0}.h5'.format(seed), 'ellipseData')
for i in range(ns):
    
    meshGMSH = meut.ellipseMesh2Domains(x0L, y0L, LxL, LyL, NL, ellipseData[i,:,:], Lxt, Lyt, lcar, x0 = x0, y0 = y0)
    meshGMSH.setTransfiniteBoundary(NpLxt)
    meshGMSH.setTransfiniteInternalBoundary(NpLxL)   
    
    meshGMSH.setNameMesh(folder + "mesh_temp_{0}.xdmf".format(seed))
    mesh = meshGMSH.getEnrichedMesh()
     
    sigma, sigmaEps = fmts.getSigma_SigmaEps(param,mesh,eps, op = 'cpp')
    
    # Solving with Multiphenics
    U = mpms.solveMultiscale(param, mesh, eps, op = opModel, others = {'polyorder' : 2, 'per': [x0, x0 + Lxt, y0, y0 + Lyt]})

    T, snap_a[i,:], snap_B[i,:,:] = feut.getAffineTransformationLocal(U[0],mesh,[0,1], justTranslation = False)    

    usol.interpolate(U[0])
   
    snap_solutions[i,:] = usol.vector().get_local()[:]
    snap_sigmas[i,:] = fmts.homogenisation(U[0], mesh, sigma, [0,1], sigmaEps).flatten()[[0,2,1]]              
    

fsnaps.close()
fellipse.close()

# for seed in range(0,20):
#     print(seed)
#     np.random.seed(seed)
#     ellipseData, PermTotal, PermBox = geni.circularRegular2Regions(r0, r1, NxL, NyL, Lx, Ly, maxOffset, ordered = False)
#     ellipseData = ellipseData[PermTotal]
    
#     ellipseData[:,2] = enforceVfracPerOffset(ellipseData[:,2], NxL, maxOffset, H, Vfrac)
#     ellipseData[(NxL+2)**2:,2] = rm # enforcing same radius after the first after the 
    
#     np.savetxt(folder + 'ellipseData_{0}.txt'.format(seed), ellipseData)
    
#     # for offset in range(Offset0,maxOffset + 1):
#     for offset in range(Offset0,maxOffset): # just temporary to keep the same random pattern, but dont generate the last offset
                     
#         Nt = (NxL + 2*offset)**2
#         Lxt = Lyt =  H*np.sqrt(Nt)
#         NpLxt = int(Lxt/lcar) + 1
         
#         x0 = x0L - offset*H; y0 = y0L - offset*H
                
#         if(offset == 0):
#             meshGMSH = meut.ellipseMesh2(ellipseData[:Nt], x0 = x0L, y0 = x0L, Lx = LxL , Ly = LxL , lcar = lcar)
#             meshGMSH.setTransfiniteBoundary(NpLxL)
        
#         else:
#             meshGMSH = meut.ellipseMesh2Domains(x0L, y0L, LxL, LyL, NL, ellipseData[:Nt], Lxt, Lyt, lcar, x0 = x0, y0 = y0)
#             meshGMSH.setTransfiniteBoundary(NpLxt)
#             meshGMSH.setTransfiniteInternalBoundary(NpLxL)   

#         meshGMSH.setNameMesh(folder + "mesh_offset{0}_{1}.{2}".format(offset,seed,'xdmf'))
#         mesh = meshGMSH.getEnrichedMesh() 


# os.system("rm " + radFile.format('solRed_{0}_offset{1}_{2}'.format(opModel,offset,seed),'h5'))

# with HDF5File(MPI.comm_world, radFile.format('solRed_{0}_offset{1}_{2}'.format(opModel,offset,seed),'h5'), 'w') as f:
#     f.write(U[0], 'basic')
        
# iofe.postProcessing_complete(U[0], folder + 'sol_temp.xdmf', ['u','lame','vonMises'], param)       



