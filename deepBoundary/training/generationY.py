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

from mpi4py import MPI as pyMPI
import pickle

comm = MPI.comm_world

normL2 = lambda x,dx : np.sqrt(assemble(inner(x,x)*dx))


folder = "/Users/felipefr/EPFL/newDLPDES/DATA/deepBoundary/runAgain/data1/"
radFile = folder + "RVE_POD_{0}.{1}"

BC = 'periodic'
BCtest = 'Per'

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

contrast = 10.0
E2 = 1.0
E1 = contrast*E2 # inclusions
nu1 = 0.3
nu2 = 0.3

ns = 400

Nbasis = np.array([4,8,12,18,25,50,75,100,125,150,156])
Nmax = 10

Ntest = 400
NNbasis = len(Nbasis) 

EpsFluc = np.loadtxt(folder + 'Eps{0}.txt'.format(BCtest))
StressList = np.loadtxt(folder + 'sigmaList.txt')

S = []
for i in range(ns):    
    print('loading simulation ' + str(i) )
    mesh = fela.EnrichedMesh(radFile.format('reduced_' + str(i),'xml'))
    V = VectorFunctionSpace(mesh,"CG", 1)
    utemp = Function(V)
        
    with HDF5File(comm, radFile.format('solution_red_' + str(i),'h5'), 'r') as f:
        f.read(utemp, 'basic')
        S.append(utemp)
        
    del utemp, V, mesh


C = np.loadtxt(folder + 'C.txt')
C = C/ns

# np.savetxt(folder + 'C.txt',C)
sig, U = np.linalg.eig(C)

asort = np.argsort(sig)
sig = sig[asort[::-1]]
U = U[:,asort[::-1]]

g = gmts.displacementGeneratorBoundary(x0L,y0L, LxL, LyL, NpLxL)

Wbasis = np.zeros((2*g.npoints,Nmax))
WbasisFunc = []


for i in range(Nmax):
    print("computing basis " + str(i))
    V = S[i].function_space()
    mesh = V.mesh()
    utemp = Function(V)
    
    for j in range(ns):
        utemp.vector()[:] = utemp.vector()[:] + (U[j,i]/np.sqrt(ns))*interpolate(S[j],V).vector()[:]     
    
    WbasisFunc.append(utemp)
    Wbasis[:,i] = g(mesh,utemp).flatten()


mu1 = elut.eng2mu(nu1,E1)
lamb1 = elut.eng2lambPlane(nu1,E1)
mu2 = elut.eng2mu(nu2,E2)
lamb2 = elut.eng2lambPlane(nu2,E2)

eps0 = np.zeros((2,2))
eps0[0,0] = 0.1
eps0 = 0.5*(eps0 + eps0.T)

param = np.array([[lamb1, mu1], [lamb2,mu2]])    
bbasis = []

for j in range(Nmax):
    bbasis.append(fmts.PointExpression(Wbasis[:,j], g))

Nmax = 50
YlistFunc = np.zeros((Ntest,Nmax))
Ylist = np.zeros((Ntest,Nmax))


for nt in range(Ntest):
    print(".... Now computing test ", nt)
    # Generating reference solution
    ellipseData = np.loadtxt(folder + 'ellipseData_' + str(nt) + '.txt')[:NL]
 
    meshGMSH = gmsh.ellipseMesh2(ellipseData, x0L, y0L, LxL, LyL, lcar, ifPeriodic)
    meshGMSH.setTransfiniteBoundary(NpLxL)
    
    mesh = feut.getMesh(meshGMSH, '_reference', radFile)
    meshRef = refine(refine(mesh))
    
    BM = fmts.getBMrestriction(g, mesh)        
    UL_ref_ = g(mesh,S[nt])
    
    UL_ref = fmts.PointExpression(UL_ref_, g, 'python') 
    # errorMR = BM.T @ UL_ref_
    # print('errorMR' , errorMR)
    
    epsL = eps0 + EpsFluc[nt,:].reshape((2,2))
    
    Uref = mpms.solveMultiscale(param, mesh, epsL, op = 'BCdirich_lag', others = [UL_ref])[0]
    V = Uref.function_space()
    Vref = VectorFunctionSpace(meshRef,"CG", 1)
    Urefref = interpolate(Uref,Vref)
    # sigmaL, sigmaEpsL = fmts.getSigma_SigmaEps(param,mesh,epsL)
     
    # sigma_ref[nt,:] = fmts.homogenisation(Uref, mesh, sigmaL, [0,1], sigmaEpsL).flatten()
    
    # print("Checking homogenisation : ", sigma_ref[nt,:], StressList[nt,:])
    
    # end mesh creation
    
    alpha = np.zeros(Nmax)
    alphaFunc = np.zeros(Nmax)
    
    meas_ds = assemble( Constant(1.0)*mesh.ds ) 
    WbasisFuncProj = []
    
    dxRef = Measure('dx', meshRef) 
    for j in range(Nmax):
        alpha[j] = assemble( inner(bbasis[j], Uref)*mesh.ds )/meas_ds
        WbasisFuncProj.append(interpolate(WbasisFunc[j],Vref))
        alphaFunc[j] = assemble( inner( grad(WbasisFuncProj[-1]), grad(Urefref))*dxRef )
    
    
    YlistFunc[nt,:] = alphaFunc[:]
    Ylist[nt,:] = alpha[:]
    
np.savetxt("Y2.txt", Ylist)
np.savetxt("Y2Func.txt", YlistFunc)

