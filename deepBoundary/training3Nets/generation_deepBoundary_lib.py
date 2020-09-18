import sys, os
from numpy import isclose
import fenics
from dolfin import *
import matplotlib.pyplot as plt
from multiphenics import *
sys.path.insert(0, '../../utils/')

import dill

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
import ioFenicsWrappers as iofe

from timeit import default_timer as timer

from mpi4py import MPI as pyMPI
import pickle
from commonParameters import *
import h5py 
from myCoeffClass import *

comm = MPI.comm_world

defaultCompression = {'dtype' : 'f8',  'compression' : "gzip", 
                           'compression_opts' : 1, 'shuffle' : False}

ten2voigt = lambda A : np.array([A[0,0],A[1,1],0.5*(A[0,1] + A[1,0])])

def recoverFenicsSimulation(nameMesh, nameSolution): 
    print('loading simulation ' + nameMesh + ' ' + nameSolution)
    utemp = Function(VectorFunctionSpace(fela.EnrichedMesh(nameMesh),"CG", 1))

    with HDF5File(comm, nameSolution , 'r') as f:
        f.read(utemp, 'basic')
    
    return utemp

loadSimulation = lambda i,rf,rs : recoverFenicsSimulation(rf.format(i,'xml'), rs.format(i,'h5'))

def loadSimulations(ib,ie, radFile, radSolution):
    S = {}
    for i in range(ib,ie):    
        S[str(i)] = loadSimulation(i, radFile, radSolution)
        
    return S

def interpolationSolutions(Isol,Vref,ns,radFile, radSolution, N0 = 0):
        
    for i,ii in enumerate(range(N0,ns)):
        ui = loadSimulation(ii,radFile, radSolution)
        Isol[i,:] = interpolate(ui,Vref).vector()[:]

    
def getBlockCorrelation(A,N, Nblocks, dotProduct, radFile, radSolution, Vref, dxRef, division = False, N0 = 0):
   
    p = np.linspace(N0,N,Nblocks+1).astype('int')
    
    for i in range(Nblocks):
        
        S = loadSimulations(p[i], p[i+1], radFile, radSolution)
        for j in range(i,Nblocks):     
            
            for jj in range(p[j], p[j+1]):     ## think in passing it to above
                Sj = S[str(jj)] if i==j else loadSimulations(jj, radFile, radSolution)
                
                ISj = interpolate(Sj,Vref)
                
                for ii in range(p[i], min(jj+1,p[i+1])):                                    
                    ISi = interpolate(S[str(ii)],Vref)
                    print("assembling i,j", ii, jj )
                    A[ii,jj] = dotProduct(ISi,ISj,dxRef)
                    A[jj,ii] = A[ii,jj]


    # A *= N**-1.
    if(division):
        A[:,:] = (1./N)*A[:,:]
    
def getCorrelation_fromInterpolation(A,N, Isol, dotProduct, radFile, radSolution, Vref, dxRef, division = False, N0 = 0):
    
    ui = Function(Vref)
    uj = Function(Vref)
    for i in range(N0, N):     ## think in passing it to above
        ui.vector().set_local(np.array(Isol[i,:]))
        for j in range(i, N):  
            print("assembling i,j", i, j )
            uj.vector().set_local(np.array(Isol[j,:]))               
            A[i,j] = dotProduct(ui,uj,dxRef)
            A[j,i] = A[i,j]


    # A *= N**-1.
    if(division):
        A[:,:] = (1./N)*A[:,:]



# Computing basis 
def computingBasis(Wbasis,C,Isol,Nmax,radFile, radSolution, divisionInC = False, N0=0,N1=0):
    
    if(N1 ==0):
        N1 = Nmax
        
    sig, U = np.linalg.eigh(C)
   
    asort = np.argsort(sig)
    sig = sig[asort[::-1]]
    U = U[:,asort[::-1]]
    
    
    ns = len(C)
    
    if(divisionInC):
        fac = np.sqrt(ns) 
    else:
        fac = float(ns)    
    
    for i in range(N0,N1):
        print("computing basis " , i )
    
        Wbasis[i,:] = (U[:,i]/(fac*np.sqrt(sig[i]))).reshape((1,len(sig)))@Isol
        # (U[:,:Nmax]/np.sqrt(ns*sig[:Nmax])).T@Isol


#  ================  Extracting Alphas ============================================
def getAlphas(Ylist, Wbasis,Isol,ns,Nmax,radFile, radSolution, dotProduct, Vref, dxRef):   # fill Ylist = np.zeros((ns,Nmax)) 
    basis = Function(Vref)
    usol = Function(Vref) 
    
    for i in range(ns):
        print(".... Now computing training ", i)
        usol.vector().set_local(np.array(Isol[i,:]))
    
        for j in range(Nmax):
            basis.vector().set_local(np.array(Wbasis[j,:]))
            Ylist[i,j] = dotProduct(basis, usol, dxRef)


def getProjection(alpha_u,Wbasis,Vref):
    Proj_u = Function(Vref)
    Proj_u.vector().set_local(Wbasis.T @ alpha_u)
    return Proj_u

def getErrors(N,Ylist,Wbasis, Isol, Vref, dxRef, dotProduct):
    ns = len(Ylist)
    ei = Function(Vref)
    errors = np.zeros(ns)
    for i in range(ns):
        ei.vector().set_local(Isol[i,:] - Wbasis[:N,:].T@Ylist[i,:N] )
        errors[i] = np.sqrt( dotProduct(ei,ei,dxRef) )
        
    return errors

def getMSE(NbasisList,Ylist,Wbasis, Isol, Vref, dxRef, dotProduct):
    MSE = []
    ns = len(Ylist)
    for N in NbasisList:
        print("computing mse error for N=", N)
        errors = getErrors(N,Ylist,Wbasis, Isol, Vref, dxRef, dotProduct)
        MSE.append(np.sqrt(np.sum(errors*errors)/ns))
        
    return np.array(MSE)
    
def getMSEstresses(NbasisList,Ylist,tau,tau0,sigmaList):
    MSE = []
    ns = len(sigmaList)
    
    # tau = tau[:ns,:3*NbasisList[-1]].reshape((ns,NbasisList[-1],3))
    for N in NbasisList:
        print("computing mse error for N=", N)        
        stressR = np.einsum('ijk,ij->ik',tau[:ns,:N,:],Ylist[:ns,:N]) + tau0[:ns]
        # print(stressR - sigmaList) 
        errors = np.linalg.norm( sigmaList - stressR, axis = 1)
        MSE.append(np.sqrt(np.sum(errors*errors)/ns))
        
    return np.array(MSE)

def getStressBasis(tau, Wbasis, ns, Nmax, EpsFlucPrefix, nameMeshPrefix, Vref, param, EpsDirection, op = "direct"): # tau = [ tau_basis, tau_0]
    
    contrast = param[2]
    E1 = param[0]
    E2 = contrast*E1 # inclusions
    nu1 = param[1]
    nu2 = param[1]
    
    mu1 = elut.eng2mu(nu1,E1)
    lamb1 = elut.eng2lambPlane(nu1,E1)
    mu2 = elut.eng2mu(nu2,E2)
    lamb2 = elut.eng2lambPlane(nu2,E2)
    
    param = np.array([[lamb1, mu1], [lamb2,mu2],[lamb1, mu1], [lamb2,mu2]])
    
    EpsUnits = np.array([[1.,0.,0.,0.], [0.,0.,0.,1.],[0.,0.5,0.5,0.]])[EpsDirection,:]
    
    if(len(EpsFlucPrefix)> 0):
        EpsFluc = np.loadtxt(EpsFlucPrefix.format(EpsDirection))
    else:
        EpsFluc = np.zeros((ns,4))
    
    basis = Function(Vref)
    
    if(op == 'direct'):
        transformBasis = lambda w,W, epsbar : interpolate(w,W)
    elif(op == 'solvePDE_BC'):
        transformBasis = lambda w,W, epsbar : mpms.solveMultiscale(param[0:2,:], W.mesh(), epsbar, op = 'BCdirich_lag', others = [w])[0]
    else:
        print('Wrong option ', op)
        input()
    
    for i in range(ns):
        print(".... Now computing tau for test ", i)
        
        mesh = fela.EnrichedMesh(nameMeshPrefix.format(i,'xml'))
        V = VectorFunctionSpace(mesh,"CG", 1)
        
        epsL = (EpsUnits + EpsFluc[i,:]).reshape((2,2))
        sigmaL, sigmaEpsL = fmts.getSigma_SigmaEps(param,mesh,epsL)
        sigmaL0, sigmaEpsL0 = fmts.getSigma_SigmaEps(param,mesh,np.zeros((2,2)))
        
        basis.vector().set_local(np.zeros(Vref.dim()))
        Ibasis = transformBasis(basis,V,epsL)
        tau[1][i,:] = ten2voigt(fmts.homogenisation(Ibasis, mesh, sigmaL, [0,1], sigmaEpsL))
        
        for j in range(Nmax):
            basis.vector().set_local(Wbasis[j,:])
            Ibasis = transformBasis(basis,V, np.zeros((2,2)))
            tau[0][i,j,:] = ten2voigt(fmts.homogenisation(Ibasis, mesh, sigmaL0, [0,1], sigmaEpsL0))



def getStressBasis_Vrefbased(tau, Wbasis, ns, Nmax, EpsFlucPrefix, nameMeshPrefix, Vref, param, EpsDirection): # tau = [ tau_basis, tau_0]
    
    contrast = param[2]
    E1 = param[0]
    E2 = contrast*E1 # inclusions
    nu1 = param[1]
    nu2 = param[1]
    
    mu1 = elut.eng2mu(nu1,E1)
    lamb1 = elut.eng2lambPlane(nu1,E1)
    mu2 = elut.eng2mu(nu2,E2)
    lamb2 = elut.eng2lambPlane(nu2,E2)
    
    param = np.array([[lamb1, mu1], [lamb2,mu2]])
       
    EpsUnits = np.array([[1.,0.,0.,0.], [0.,0.,0.,1.],[0.,0.5,0.5,0.]])[EpsDirection,:]
    
    if(len(EpsFlucPrefix)> 0):
        EpsFluc = np.loadtxt(EpsFlucPrefix.format(EpsDirection))
    else:
        EpsFluc = np.zeros((ns,4))
    
    basis = Function(Vref)
    VSref = FunctionSpace(Vref.mesh(), 'CG', 4)
    dxRef = Measure('dx', Vref.mesh()) 
    vol = assemble(Constant(1.0)*dxRef)
    
    for i in range(ns):
        print(".... Now computing tau for test ", i)
        
        mesh = fela.EnrichedMesh(nameMeshPrefix.format(i,'xml'))
        V = VectorFunctionSpace(mesh,"CG", 1)
        VS  = FunctionSpace(mesh, 'DG', 0)
        
        materials = mesh.subdomains.array().astype('int32')
        materials -= np.min(materials)
        lame = getMyCoeff(materials , param, op = 'python') 
        
        lame0 = interpolate(iofe.local_project(lame[0],VS), VSref)
        lame1 = interpolate(iofe.local_project(lame[1],VS), VSref)
        lame_ = as_vector((lame0,lame1))
        
        sigma = lambda u: fela.sigmaLame(u,lame_)        
        
        epsL = (EpsUnits + EpsFluc[i,:]).reshape((2,2))
        sigmaL, sigmaEpsL = fmts.getSigma_SigmaEps(param,mesh,epsL)
        
        basis.vector().set_local(np.zeros(Vref.dim()))
        Ibasis = interpolate(basis,V)
        tau[1][i,:] = ten2voigt(fmts.homogenisation(Ibasis, mesh, sigmaL, [0,1], sigmaEpsL))
        
        for j in range(Nmax):
            basis.vector().set_local(np.array(Wbasis[j,:]))
            tau[0][i,j,:] = ten2voigt(feut.Integral(sigma(basis),dxRef,(2,2)))/vol

           