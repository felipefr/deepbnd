import sys, os
from numpy import isclose
from dolfin import *
from multiphenics import *
sys.path.insert(0, '../../../utils/')

import fenicsWrapperElasticity as fela
import matplotlib.pyplot as plt
import numpy as np
import meshUtils as meut
import generationInclusions as geni
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
from skopt.sampler import Lhs

import symmetryLib as syml

def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]


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

def getScikitoptSampleVolFraction_adaptativeSample(NR,Vfrac,r0,r1, p, rm, facL, facR, M, N, 
                                                   indexes, seed, op = 'lhs'):
    space = Space(NR*[(r0, r1)])
    if(op == 'lhs'):
        sampler = Lhs(lhs_type="centered", criterion=None)
    elif(op == 'lhs_maxmin'):
        sampler = Lhs(criterion="maximin", iterations=100)
    elif(op == 'sobol'):
        sampler = Sobol()
    
    Rlist = []
    np.random.seed(seed)
    
    for pi in range(p**M):
        pibin = bin(pi)[2:] # supposing p = 2
        pibin = pibin + (4-len(pibin))*'0' # to complete 4 digits
        pibin = [int(pibin[i]) for i in range(M)]
        
        Rlist.append( np.array(sampler.generate(space.dimensions, N)) )
        
        for j, jj in enumerate(indexes):
            k = pibin[j]
            if(k == 0):
                Rlist[-1][:,jj] = r0 + facL*( Rlist[-1][:,jj] - r0 )
            elif(k == 1):
                Rlist[-1][:,jj] = rm + facR*( Rlist[-1][:,jj] - r0 )
    
        R = np.concatenate(Rlist,axis = 0)
    
    for i in range(N*p**M): # total samples, impose volume fraction in the whole volume (maybe it should be partionated)
        alphaFrac = H*np.sqrt(NR*Vfrac/(np.pi*np.sum(R[i,:]**2)))
        R[i,:] *= alphaFrac
    

    return R


def getScikitoptSampleVolFraction_adaptativeSample_frozen(NR,Vfrac,r0,r1, H, p, M, N, 
                                                   indexes, seed, op = 'lhs'):
    space = Space(NR*[(r0, r1)])
    if(op == 'lhs'):
        sampler = Lhs(lhs_type="centered", criterion=None)
    elif(op == 'lhs_maxmin'):
        sampler = Lhs(criterion="maximin", iterations=20)
    elif(op == 'sobol'):
        sampler = Sobol()
    
    Rlist = []
    np.random.seed(seed)
    
        
    rlim = [r0 + i*(r1-r0)/p for i in range(p+1)]
    faclim = [(rlim[i+1]-rlim[i])/(r1-r0) for i in range(p)]
    
    for pi in range(p**M):
        pibin = numberToBase(pi,p) # supposing p = 2
        pibin = pibin + (4-len(pibin))*[0] # to complete 4 digits
        
        Rlist.append( np.array(sampler.generate(space.dimensions, N)) )
        
        for j in range(len(Rlist[-1][0,:])):
            if(j not in indexes):
                Rlist[-1][:,j] = Rlist[0][:,j]
        
        for j, jj in enumerate(indexes): #j = 0,1,2,..., jj = I_0,I_1,I_2,...
            k = pibin[j]
            Rlist[-1][:,jj] = rlim[k] + faclim[k]*( Rlist[-1][:,jj] - r0 )
            
    
        R = np.concatenate(Rlist,axis = 0)
    
    # for i in range(N*p**M): # total samples, impose volume fraction in the whole volume (maybe it should be partionated)
    #     alphaFrac = H*np.sqrt(NR*Vfrac/(np.pi*np.sum(R[i,:]**2)))
    #     R[i,:] = alphaFrac*R[i,:]

    for i in range(N*p**M):
        R[i,:] = enforceVfracPerOffset(R[i,:], 2, 2, H, Vfrac) # radius should be ordened interior to exterior, 
    
    
    return R


opModel = 'periodic'
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
eps = np.zeros((2,2))
eps[0,1] = 0.5
eps[1,0] = 0.5

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
usol = [Function(Vref), Function(Vref),Function(Vref),Function(Vref)]

nxy = 100
Mref_total = RectangleMesh(Point(x0, y0), Point(x0+Lxt, y0+Lyt), nxy, nxy, "right/left")
Vref_total = VectorFunctionSpace(Mref_total,"CG", 2)

usol_total = [Function(Vref_total), Function(Vref_total),Function(Vref_total),Function(Vref_total)]

M = 1
NR1 = 4
NR2 = 12
NR3 = 20
NR = NR1 + NR2 + NR3

ns = 1

p = 1
N = int(ns/(p**M))
       
ns = N*p**M
print(ns)

# Radius Generation
seed = 17 # for the test   

np.random.seed(seed)

ellipseData, PermTotal, PermBox = geni.circularRegular2Regions(r0, r1, NxL, NyL, Lxt, Lyt, maxOffset, ordered = False, x0 = x0, y0 = y0)
ellipseData = ellipseData[PermTotal]

R = getScikitoptSampleVolFraction_adaptativeSample_frozen(NR,Vfrac,r0,r1, H, p, M, N, 
                                                    [i for i in range(M)], seed, op = 'lhs_maxmin')

ellipseData[:,2] = enforceVfracPerOffset(R[0,:], NxL, maxOffset, H, Vfrac)
 
ellipseData_sym = syml.createSymmetricEllipseData(ellipseData[np.newaxis,:,:], useFiles = False)

solU = []

for i in range(4):
    print("Solving snapshot")
    start = timer()
    meshGMSH = meut.ellipseMesh2Domains(x0L, y0L, LxL, LyL, NL, ellipseData_sym[i,:,:], Lxt, Lyt, lcar, x0 = x0, y0 = y0)
    meshGMSH.setTransfiniteBoundary(NpLxt)
    meshGMSH.setTransfiniteInternalBoundary(NpLxL)   
        
    meshGMSH.setNameMesh("mesh_temp.xdmf")
    mesh = meshGMSH.getEnrichedMesh()
     
    sigmaLaw, sigmaLawEps = fmts.getSigma_SigmaEps(param,mesh,eps, op = 'cpp')
    
    # Solving with Multiphenics
    others = {'method' : 'default', 'polyorder' : 2, 'per': [x0, x0 + Lxt, y0, y0 + Lyt]}
    U = mpms.solveMultiscale(param, mesh, eps, op = opModel, others = others)
    
    T, a, B = feut.getAffineTransformationLocal(U[0],mesh,[0,1], justTranslation = False)    
    
    usol[i].interpolate(U[0])
    usol_total[i].interpolate(U[0])

    sigma = fmts.homogenisation(U[0], mesh, sigmaLaw, [0,1], sigmaLawEps).flatten()[[0,3,2]]    
    sigmasT = fmts.homogenisation(U[0], mesh, sigmaLaw, [0,1,2,3], sigmaLawEps).flatten()[[0,3,2]]      
    end = timer()
    
    print("concluded in ", end - start)      
        
    iofe.postProcessing_complete(U[0], 'sol_mesh_{0}.xdmf'.format(i), ['u','lame','vonMises'], param)    

    solU.append(U[0])
    


T_MH = Expression(('-x[0]','x[1]'), degree = 2)
T_MV = Expression(('x[0]','-x[1]'), degree = 2)
T_MD = Expression(('-x[0]','-x[1]'), degree = 2)

Transformations = [T_MH,T_MV,T_MD]
usol_total_mirror = [Function(Vref_total),Function(Vref_total),Function(Vref_total)]
usol_mirror = [Function(Vref),Function(Vref),Function(Vref)]

for i,T in enumerate(Transformations):
    usol_total_mirror[i].interpolate(feut.myfog(usol_total[0],T)) 
    usol_mirror[i].interpolate(feut.myfog(usol[0],T)) 

dx = Measure('dx',Mref_total)
ds = Measure('ds', Mref)

error_total = []
error = []
for i in range(3):
    error_total.append( assemble( inner(usol_total_mirror[i] - usol_total[0], usol_total_mirror[i] - usol_total[0])*dx ) )
    error.append( assemble( inner(usol_mirror[i] - usol[0], usol_mirror[i] - usol[0])*ds ) )
    

plt.figure(1)
plot(usol_total[0][1]+usol_total_mirror[2][1] , mode ='color')
plt.colorbar()
plt.show()

