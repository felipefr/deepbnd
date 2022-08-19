import sys, os
from numpy import isclose
import numpy as np
import fenics
from dolfin import *
import matplotlib.pyplot as plt
from multiphenics import *
from timeit import default_timer as timer

from deepBND.core.fenics_tools.enriched_mesh import EnrichedMesh 
import deepBND.core.data_manipulation.wrapper_h5py as myhd


defaultCompression = {'dtype' : 'f8',  'compression' : "gzip", 
                           'compression_opts' : 1, 'shuffle' : False}

ten2voigt = lambda A : np.array([A[0,0],A[1,1],0.5*(A[0,1] + A[1,0])])

def recoverFenicsSimulation(nameMesh, nameSolution): 
    print('loading simulation ' + nameMesh + ' ' + nameSolution)
    utemp = Function(VectorFunctionSpace(EnrichedMesh(nameMesh),"CG", 1))

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

def getCorrelation_fromInterpolationMatricial(A,N, Isol, dotProduct, Vref, dxRef, division = False, N0 = 0):
    
    ui = Function(Vref)
    uj = Function(Vref)
    
    Nh = len(Isol[0,:])
    M = np.zeros((Nh,Nh))
    
    for i in range(Nh):     ## think in passing it to above
        ei = np.zeros(Nh)
        ei[i] = 1.0
        ui.vector().set_local(ei)
        for j in range(i, Nh):     ## think in passing it to above
            ej = np.zeros(Nh)
            ej[j] = 1.0
            uj.vector().set_local(ej)
            M[i,j] = dotProduct(ui,uj,dxRef)
            M[j,i] = M[i,j]
            
    S = np.array(Isol[:,:])
    A[:,:] = S@M@S.T
    
    # A *= N**-1.
    if(division):
        A[:,:] = (1./N)*A[:,:]


def getCorrelation_fromInterpolation(A,N, Isol, dotProduct, Vref, dxRef, division = False, N0 = 0):
    
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
def computingBasis(Wbasis,C,Isol,Nmax, divisionInC = False, N0=0,N1=0):
    
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
        fac = 1.0    
    
    for i in range(N0,N1):
        print("computing basis " , i )
    
        Wbasis[i,:] = (U[:,i]/(fac*np.sqrt(sig[i]))).reshape((1,len(sig)))@Isol
        # (U[:,:Nmax]/np.sqrt(ns*sig[:Nmax])).T@Isol

    return sig, U

def getMassMatrix(Vref,dxRef,dotProduct):
    u = TrialFunction(Vref)
    v = TestFunction(Vref)

    a = dotProduct(u,v,dxRef)
    A = assemble(a)
    
    return A.array()

def computingBasis_svd(Wbasis, M, Isol, Nmax, Vref, dxRef, dotProduct):
    M[:,:] = getMassMatrix(Vref,dxRef,dotProduct)
    print("Mass matrix built")
    UM,SM,VMT = np.linalg.svd(M)  
    print("Mass matrix factorised")
    Msqrt = (UM[:,:len(SM)]*np.sqrt(SM))@VMT
            
    U, sig, VT = np.linalg.svd(Isol@Msqrt,full_matrices = False) # sig here means the sqrt of eigenvalues for the correlation matrix method
    
    print("SVD on the snapshots perfomed")
    
    print(U.shape)
    print(sig.shape)
    print(Isol.shape)
    
    U = U[:,:len(sig)]
    ns = len(U)

    for i in range(Nmax):
        print("computing basis " , i )
        Wbasis[i,:] = (U[:,i]/sig[i]).reshape((1,ns))@Isol
        
        
    return sig**2, U    


def zerofyDummyDofsBasis(Wbasis, M):
    mask = np.sum(np.abs(M), axis = 1).flatten()
    mask[mask.nonzero()[0]] = 1.0
    
    for i in range(Wbasis.shape[0]):
        Wbasis[i,:] = mask*Wbasis[i,:] 
        

def test_zerofiedBasis(Wbasis0_name, Wbasis_name):
    
    Wbasis_S = myhd.loadhd5( Wbasis_name, 'Wbasis_S')
    Wbasis_A = myhd.loadhd5( Wbasis_name, 'Wbasis_A')
    
    Wbasis0_S = myhd.loadhd5( Wbasis0_name, 'Wbasis_S')
    Wbasis0_A = myhd.loadhd5( Wbasis0_name, 'Wbasis_A')
    
    M = myhd.loadhd5( Wbasis_name, 'massMat')
    
    mask = np.sum(np.abs(M), axis = 1).flatten()
    i_nz = mask.nonzero()[0]
    
    assert np.allclose(Wbasis_S[:,i_nz], Wbasis0_S[:,i_nz] )
    assert np.allclose(Wbasis_A[:,i_nz], Wbasis0_A[:,i_nz] )
    
    return Wbasis_S[:,i_nz], Wbasis0_S[:,i_nz], M

def reinterpolateWbasis(Wbasis, Vref, Wbasis0 , Vref0):
    
    Nmax = len(Wbasis0)

    basis = Function(Vref0)

    for i in range(Nmax):
        basis.vector().set_local(Wbasis0[i,:])
        Wbasis[i,:] = interpolate(basis,Vref).vector().get_local()[:]



#  ================  Extracting Alphas ============================================
def getAlphas(Ylist, Wbasis,Isol,ns,Nmax, dotProduct, Vref, dxRef):   # fill Ylist = np.zeros((ns,Nmax)) 
    basis = Function(Vref)
    usol = Function(Vref) 
    
    for i in range(ns):
        print(".... Now computing training ", i)
        usol.vector().set_local(np.array(Isol[i,:]))
    
        for j in range(Nmax):
            basis.vector().set_local(np.array(Wbasis[j,:]))
            Ylist[i,j] = dotProduct(basis, usol, dxRef)
            
def getAlphas_fast(Wbasis_M,Isol,ns,Nmax, dotProduct, Vref, dxRef):   # fill Ylist = np.zeros((ns,Nmax)) 
    Wbasis, M = Wbasis_M
    return Isol @ M @ Wbasis[:Nmax,:].T


def getProjection(alpha_u,Wbasis,Vref):
    Proj_u = Function(Vref)
    Proj_u.vector().set_local(Wbasis.T @ alpha_u)
    return Proj_u

# Compute error L2 given the projections and base for each snapshot (dimension of Y dictates dimensions)
def getErrors(Ylist,Wbasis, Isol, Vref, dxRef, dotProduct):
    ns , N = Ylist.shape
    assert(Wbasis.shape[0] == N)
    
    ei = Function(Vref)
    errors = np.zeros(ns)
    for i in range(ns):
        ei.vector().set_local(Isol[i,:] - Wbasis.T@Ylist[i,:] )
        errors[i] = np.sqrt( dotProduct(ei,ei,dxRef) )
        
    return errors

# Compute MSE error in the L2 norm : (POD or total error)
def getMSE(NbasisList, Ylist,Wbasis, Isol, Vref, dxRef, dotProduct):
    MSE = []
    ns = len(Ylist)
    for N in NbasisList:
        print("computing mse error for N=", N)
        errors = getErrors(Ylist[:,:N],Wbasis[:N,:], Isol, Vref, dxRef, dotProduct)
        MSE.append(np.sum(errors**2)/ns)
        
    return np.array(MSE)

def getMSE_fast(NbasisList,Ylist,Wbasis_M, Isol):
    MSE = np.zeros(len(NbasisList))
    Wbasis, M = Wbasis_M
    ns = len(Ylist)
    for k, N in enumerate(NbasisList):
        print("computing mse error for N=", N)
        E = Isol - Ylist[:,:N] @ Wbasis[:N,:]
        error = np.array([np.dot(E[i,:],M@E[i,:]) for i in range(len(E))])
        MSE[k] = np.sum(error/ns)
        
    return MSE


def getMSE_DNN_fast(NbasisList,Ylist1, Ylist2,Wbasis_M):
    MSE = np.zeros(len(NbasisList))
    Wbasis, M = Wbasis_M
    ns = len(Ylist1)
    for k, N in enumerate(NbasisList):
        print("computing mse error for N=", N)
        E = Ylist1[:,:N] @ Wbasis[:N,:] - Ylist2[:,:N] @ Wbasis[:N,:]
        error = np.array([np.dot(E[i,:],M@E[i,:]) for i in range(len(E))])
        MSE[k] = np.sum(error/ns)
        
    return MSE



# Compute the error in the projections simply
def getMSE_DNN(Yp,Yt):
    ns = len(Ylist)
    ei = Function(Vref)
    errors = np.zeros(ns)
    for i in range(ns):
        ei.vector().set_local(Isol[i,:] - Wbasis[:N,:].T@Ylist[i,:N] )
        errors[i] = np.sqrt( dotProduct(ei,ei,dxRef) )
        
    return errors


    
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

            
            
            

            
           
    
    



