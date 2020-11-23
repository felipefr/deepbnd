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
from timeit import default_timer as timer

from skopt.space import Space
from skopt.sampler import Sobol
from skopt.sampler import Lhs
from skopt.sampler import Halton
from skopt.sampler import Hammersly
from skopt.sampler import Grid
from scipy.spatial.distance import pdist

def enforceVfracPerOffset(radius, NxL, maxOffset, H, Vfrac): # radius should be ordened interior to exterior, 
    for i in range(maxOffset+1):
        ni =  (NxL + 2*(i-1))**2 
        nout = (NxL + 2*i)**2
        alphaFrac = H*np.sqrt((nout-ni)*Vfrac/(np.pi*np.sum(radius[ni:nout]**2)))
        radius[ni:nout] *= alphaFrac
        
    return radius


def getUniformSampleVolFraction(n,NR,Vfrac,r0,r1):
    R = np.zeros((n,NR))
    for i in range(n):
        R[i,:] = r0 + (r1 - r0)*np.random.rand(NR)
        alphaFrac = H*np.sqrt(NR*Vfrac/(np.pi*np.sum(R[i,:]**2)))
        R[i,:] *= alphaFrac
    
    return R

def getScikitoptSampleVolFraction(n,NR,Vfrac,r0,r1, op = 'lhs'):
    space = Space(NR*[(r0, r1)])
    if(op == 'lhs'):
        sampler = Lhs(lhs_type="centered", criterion=None)
    elif(op == 'lhs_maxmin'):
        sampler = Lhs(criterion="maximin", iterations=100)
    elif(op == 'sobol'):
        sampler = Sobol()
        
    R = np.array(sampler.generate(space.dimensions, n))
    print(R.shape)
    for i in range(n):
        alphaFrac = H*np.sqrt(NR*Vfrac/(np.pi*np.sum(R[i,:]**2)))
        R[i,:] *= alphaFrac
    
    return R

folder = ["/Users", "/home"][0] + "/felipefr/switchdrive/scratch/deepBoundary/smartGeneration/LHS_maxmin_full/"

radFile = folder + "RVE_POD_{0}.{1}"

opModel = 'periodic'
opLoad = 'shear'
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

maxOffset = 2

H = 1.0 # size of each square
NxL = NyL = 2
NL = NxL*NyL
x0L = y0L = -H 
LxL = LyL = 2*H
lcar = (2/30)*H
Nx = (NxL+2*maxOffset)
Ny = (NyL+2*maxOffset)
Lxt = Nx*H
Lyt = Ny*H
NpLxt = int(Lxt/lcar) + 1
NpLxL = int(LxL/lcar) + 1
print("NpLxL=", NpLxL) 
x0 = -Lxt/2.0
y0 = -Lyt/2.0
r0 = 0.2*H
r1 = 0.4*H
Vfrac = 0.282743
rm = H*np.sqrt(Vfrac/np.pi)

meshRef = meut.degeneratedBoundaryRectangleMesh(x0 = x0L, y0 = y0L, Lx = LxL , Ly = LyL , Nb = 21)
meshRef.generate()
meshRef.write('boundaryMesh.xdmf', 'fenics')
Mref = meut.EnrichedMesh('boundaryMesh.xdmf')
Vref = VectorFunctionSpace(Mref,"CG", 1)
usol = Function(Vref)

ns1 = 100
ns2 = 10
ns3 = 10

NR1 = 4
NR2 = 12
NR3 = 20
NR = NR1 + NR2 + NR3

ns = ns1*ns2*ns3

# Radius Generation
seed = 1
np.random.seed(seed)

os.system('rm ' + folder +  'ellipseData_{0}.h5'.format(seed))
X, f = myhd.zeros_openFile(filename = folder +  'ellipseData_{0}.h5'.format(seed),  shape = (ns,Ny*Nx,5), label = 'ellipseData', mode = 'w-')

ellipseData, PermTotal, PermBox = geni.circularRegular2Regions(r0, r1, NxL, NyL, Lxt, Lyt, maxOffset, ordered = False, x0 = x0, y0 = y0)
ellipseData = ellipseData[PermTotal]

# R1 = getUniformSampleVolFraction(ns1,NR1,Vfrac,r0,r1)
# R2 = getUniformSampleVolFraction(ns2,NR2,Vfrac,r0,r1)
# R3 = getUniformSampleVolFraction(ns3,NR3,Vfrac,r0,r1)

# R1 = getScikitoptSampleVolFraction(ns1,NR1,Vfrac,r0,r1,'lhs_maxmin')
# R2 = getScikitoptSampleVolFraction(ns2,NR2,Vfrac,r0,r1,'lhs_maxmin')
# R3 = getScikitoptSampleVolFraction(ns3,NR3,Vfrac,r0,r1,'lhs_maxmin')


# for i in range(ns3):
#     ellipseData[(NxL + 2)**2:(NxL + 4)**2:,2] = R3[i,:]
#     for j in range(ns2):
#         ellipseData[NxL**2:(NxL + 2)**2:,2] = R2[j,:]
#         for k in range(ns1):
#             ijk =i*ns2*ns1 + j*ns1 + k
#             ellipseData[:NxL**2,2] = R1[k,:]
#             print("inserting on ", ijk)
#             X[ijk,:,:] = ellipseData
       

R = getScikitoptSampleVolFraction(ns,NR,Vfrac,r0,r1,'lhs_maxmin')

for i in range(ns):
    ellipseData[:,2] = enforceVfracPerOffset(R[i,:], NxL, maxOffset, H, Vfrac)
    print("inserting on ", i)
    X[i,:,:] = ellipseData


f.close()


os.system('rm ' + folder +  'snapshots_{0}.h5'.format(seed))
snapshots, fsnaps = myhd.zeros_openFile(filename = folder +  'snapshots_{0}.h5'.format(seed),  
                                        shape = [(ns,Vref.dim()),(ns,3),(ns,2),(ns,2,2)], label = ['solutions','sigma','a','B'], mode = 'w-')

snap_solutions, snap_sigmas, snap_a, snap_B = snapshots
     
ellipseData, fellipse = myhd.loadhd5_openFile(folder +  'ellipseData_{0}.h5'.format(seed), 'ellipseData')
for i in range(ns):
    print("Solving snapshot", i)
    start = timer()
    meshGMSH = meut.ellipseMesh2Domains(x0L, y0L, LxL, LyL, NL, ellipseData[i,:,:], Lxt, Lyt, lcar, x0 = x0, y0 = y0)
    meshGMSH.setTransfiniteBoundary(NpLxt)
    meshGMSH.setTransfiniteInternalBoundary(NpLxL)   
    
    meshGMSH.setNameMesh(folder + "mesh_temp_{0}.xdmf".format(seed))
    mesh = meshGMSH.getEnrichedMesh()
     
    sigma, sigmaEps = fmts.getSigma_SigmaEps(param,mesh,eps, op = 'cpp')
    
    # Solving with Multiphenics
    U = mpms.solveMultiscale(param, mesh, eps, op = opModel, others = {'polyorder' : 1, 'per': [x0, x0 + Lxt, y0, y0 + Lyt]})

    T, snap_a[i,:], snap_B[i,:,:] = feut.getAffineTransformationLocal(U[0],mesh,[0,1], justTranslation = False)    

    usol.interpolate(U[0])
   
    snap_solutions[i,:] = usol.vector().get_local()[:]
    snap_sigmas[i,:] = fmts.homogenisation(U[0], mesh, sigma, [0,1], sigmaEps).flatten()[[0,3,1]]        
    end = timer()
    print("concluded in ", end - start)      
    

fsnaps.close()
# fellipse.close()

    



