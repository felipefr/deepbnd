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
import generationInclusions as geni
import myCoeffClass as coef
import fenicsMultiscale as fmts
import elasticity_utils as elut
import fenicsWrapperElasticity as fela
import multiphenicsMultiscale as mpms
import fenicsUtils as feut
import ioFenicsWrappers as iofe
import meshUtils as meut

from timeit import default_timer as timer

from mpi4py import MPI as pyMPI
import pickle
from commonParameters import *
import h5py 
from myCoeffClass import *
import myHDF5 as myhd 

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

def getStressBasis(tau, Wbasis, ns, Nmax, EpsFlucPrefix, nameMeshPrefix, Vref, param, EpsDirection, op = "direct"): # tau = [ tau_basis, tau_0, tau_0_fluc]
    
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
        EpsFluc = myhd.genericLoadfile(EpsFlucPrefix,'EpsList')[:,2,:]
    else:
        EpsFluc = np.zeros((ns,4))
    
    basis = Function(Vref)
    
    if(op == 'direct'):
        transformBasis0 = lambda w,W, epsbar : interpolate(w,W)
        transformBasis = lambda w,W, epsbar : interpolate(w,W)
    elif(op == 'solvePDE_BC'):
        transformBasis0 = lambda w,W, epsbar : mpms.solveMultiscale(param[0:2,:], W.mesh(), epsbar, op = 'Lin')[0]
        transformBasis = lambda w,W, epsbar : mpms.solveMultiscale(param[0:2,:], W.mesh(), epsbar, op = 'BCdirich_lag', others = [w])[0]
    elif(op == 'periodic'):
        transformBasis0 = lambda w,W, epsbar : mpms.solveMultiscale(param[0:2,:], W.mesh(), epsbar, op = 'periodic', others = [1./3.,2./3.,1./3.,2./3.])[0]
        transformBasis = lambda w,W, epsbar : mpms.solveMultiscale(param[0:2,:], W.mesh(), epsbar, op = 'BCdirich_lag', others = [w])[0]
    else:
        print('Wrong option ', op)
        input()
    
    for i in range(ns):
        print(".... Now computing tau for test ", i)
        
        mesh = fela.EnrichedMesh(nameMeshPrefix.format(i,'xml'))
        V = VectorFunctionSpace(mesh,"CG", 1)
        
        basis.vector().set_local(np.zeros(Vref.dim()))
        
        
        epsL = EpsUnits.reshape((2,2))
        Ibasis = transformBasis0(basis,V,epsL)
        sigmaL, sigmaEpsL = fmts.getSigma_SigmaEps(param,mesh,epsL)
        tau[1][i,:] = ten2voigt(fmts.homogenisation(Ibasis, mesh, sigmaL, [0,1], sigmaEpsL))
        
        epsL = EpsFluc[i,:].reshape((2,2))
        Ibasis = transformBasis0(basis,V,epsL)
        sigmaL, sigmaEpsL = fmts.getSigma_SigmaEps(param,mesh,epsL)
        tau[2][i,:] = ten2voigt(fmts.homogenisation(Ibasis, mesh, sigmaL, [0,1], sigmaEpsL))
        
        sigmaL0, sigmaEpsL0 = fmts.getSigma_SigmaEps(param,mesh,np.zeros((2,2)))
        
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
        EpsFluc = myhd.genericLoadfile(EpsFlucPrefix,'EpsList')[:,2,:]
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
               
        basis.vector().set_local(np.zeros(Vref.dim()))
        Ibasis = interpolate(basis,V)
        
        epsL = EpsUnits.reshape((2,2))
        sigmaL, sigmaEpsL = fmts.getSigma_SigmaEps(param,mesh,epsL)
        tau[1][i,:] = ten2voigt(fmts.homogenisation(Ibasis, mesh, sigmaL, [0,1], sigmaEpsL))
        
        epsL = EpsFluc[i,:].reshape((2,2))
        sigmaL, sigmaEpsL = fmts.getSigma_SigmaEps(param,mesh,epsL)
        tau[2][i,:] = ten2voigt(fmts.homogenisation(Ibasis, mesh, sigmaL, [0,1], sigmaEpsL))
        
        
        for j in range(Nmax):
            basis.vector().set_local(np.array(Wbasis[j,:]))
            tau[0][i,j,:] = ten2voigt(feut.Integral(sigma(basis),dxRef,(2,2)))/vol
            
            
            
def getStressBasis_generic(tau, Wbasis, WbasisAux, ns, Nmax, EpsFlucPrefix, nameMeshPrefix, Vref, param, EpsDirection, V0 = 'VL', Orth = True): # tau = [ tau_basis, tau_0]
    
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
        EpsFluc = myhd.genericLoadfile(EpsFlucPrefix,'EpsList')[:,2,:]
    else:
        EpsFluc = np.zeros((ns,4))
    
    basis = Function(Vref)
    basisAux = Function(Vref)
    VSref = FunctionSpace(Vref.mesh(), 'CG', 4)
    dxRef = Measure('dx', Vref.mesh()) 
    vol = assemble(Constant(1.0)*dxRef)
    
    if(V0 == 'VT'):
        getu0 = lambda W, epsbar : Function(W)
        getOrth = lambda w,W : interpolate(w,W)
    elif(V0 == 'VL'):
        getu0 = lambda W, epsbar : mpms.solveMultiscale(param[0:2,:], W.mesh(), epsbar, op = 'Lin')[0]
        getOrth = lambda w,W : mpms.solveMultiscale(param[0:2,:], W.mesh(), np.zeros((2,2)), op = 'BCdirich_lag', others = [w])[0]
    elif(VL == 'VP'):
        getu0 = lambda W, epsbar : mpms.solveMultiscale(param[0:2,:], W.mesh(), epsbar, op = 'periodic', others = [1./3.,2./3.,1./3.,2./3.])[0]
        getOrth = lambda w,W : mpms.solveMultiscale(param[0:2,:], W.mesh(), np.zeros((2,2)), op = 'BCdirich_lag', others = [w])[0]
    else:
        print('Wrong option ', V0)
        input()


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
               
        epsL = EpsUnits.reshape((2,2))
        u0 = getu0(V, epsL)
        sigmaL, sigmaEpsL = fmts.getSigma_SigmaEps(param,mesh,epsL)
        tau[1][i,:] = ten2voigt(fmts.homogenisation( u0, mesh, sigmaL, [0,1], sigmaEpsL))
        
        epsL = EpsFluc[i,:].reshape((2,2))
        u0 = getu0(V, epsL)
        sigmaL, sigmaEpsL = fmts.getSigma_SigmaEps(param,mesh,epsL)
        tau[2][i,:] = ten2voigt(fmts.homogenisation( u0, mesh, sigmaL, [0,1], sigmaEpsL))
        
        sigmaL0, sigmaEpsL0 = fmts.getSigma_SigmaEps(param,mesh,np.zeros((2,2)))
                
        for j in range(Nmax):
            if(Orth):
                basisAux.vector().set_local(WbasisAux[j,:])
                Ibasis = getOrth(basisAux,V)
                tau[0][i,j,:] = ten2voigt(fmts.homogenisation(Ibasis, mesh, sigmaL0, [0,1], sigmaEpsL0))

            else:    
                basis.vector().set_local(np.array(Wbasis[j,:]))
                tau[0][i,j,:] = ten2voigt(feut.Integral(sigma(basis),dxRef,(2,2)))/vol
            
            
            

def getStressReconstructed(Wbasis, Ylist, ns, nYlist, nameMeshPrefix, Vref, param):
    
    stressR = np.zeros((ns,len(nYlist),3))
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
    
    uR = Function(Vref)
    
    auxPDEproblem = lambda w,W : mpms.solveMultiscale(param[0:2,:], W.mesh(), np.zeros((2,2)), op = 'BCdirich_lag', others = [w])[0]
    
    for i in range(ns):
        print(".... Now computing tau for test ", i)
        
        mesh = fela.EnrichedMesh(nameMeshPrefix.format(i,'xml'))
        V = VectorFunctionSpace(mesh,"CG", 1)
        
        sigmaL0, sigmaEpsL0 = fmts.getSigma_SigmaEps(param,mesh,np.zeros((2,2)))
        
        uR.vector().set_local(np.zeros(Vref.dim()))
        
        for j, nY in enumerate(nYlist):
            uR.vector().set_local(Wbasis[:nY,:].T@Ylist[i,:nY])
            uR_PDE = auxPDEproblem(uR , V) 
            stressR[i,j,:] = ten2voigt(fmts.homogenisation(uR_PDE, mesh, sigmaL0, [0,1], sigmaEpsL0))


    return stressR    

class SimulationCase:
    def __init__(self, V0, ExtOrth, RBbasis, metric):
        self.V0 = V0
        self.ExtOrth = ExtOrth
        self.RBbasis = RBbasis
        self.metric = metric

        
    def checkException(self):
        if(self.ExtOrth == 1 and self.RBbasis != 'L2bnd'):
            return False
        
        if(self.ExtOrth == 0 and self.RBbasis == 'L2bnd'):
            return False
        
        if(self.ExtOrth == 1 and self.V0 == 'M'):
            return False
        
        if(self.ExtOrth == 1 and self.V0 == 'P'):
            return False
        
        if(self.V0 == 'P' and self.RBbasis == 'H10'):
            return False
        
        return True

    def getLabel(self):
        return "{0}_{1}_{2}_{3}".format(self.V0,self.ExtOrth,self.RBbasis,self.metric)    
    

class RBsimul: # specific for the case of multiscale, generalise afterwards
    
    def __init__(self, filenameBase, param,  EpsFlucPrefix, Vref, nameMeshPrefix, Nmax , EpsDirection):
        contrast = param[2]
        E1 = param[0]
        E2 = contrast*E1 # inclusions
        nu1 = param[1]
        nu2 = param[1]
        
        mu1 = elut.eng2mu(nu1,E1)
        lamb1 = elut.eng2lambPlane(nu1,E1)
        mu2 = elut.eng2mu(nu2,E2)
        lamb2 = elut.eng2lambPlane(nu2,E2)
        
        self.param = np.array([[lamb1, mu1], [lamb2,mu2]])
        
        self.EpsUnits = np.array([[1.,0.,0.,0.], [0.,0.,0.,1.],[0.,0.5,0.5,0.]])[EpsDirection,:]
        
        if(len(EpsFlucPrefix)> 0):
            # self.EpsFluc = myhd.genericLoadfile(EpsFlucPrefix,'EpsList')[:,2,:]
            # self.EpsFluc = myhd.genericLoadfile(EpsFlucPrefix,'EpsList')[:,:]
            self.EpsFluc = myhd.loadhd5(EpsFlucPrefix,'EpsList')
        else:
            self.EpsFluc = np.zeros((ns,4))
            
        self.Vref = Vref
        self.VSref = FunctionSpace(Vref.mesh(), 'CG', 4)
        self.dxRef = Measure('dx', Vref.mesh()) 
        self.dsRef = Measure('ds', Vref.mesh()) 
        self.vol = assemble(Constant(1.0)*self.dxRef)
        self.basis = Function(Vref)
        self.urefAux = Function(Vref)
        self.urefAux2 = Function(Vref)

        self.Nmax = Nmax
        
        self.WbasisDict = {}
        self.Isol = None
        
        f = h5py.File(filenameBase, 'a')
        
        self.filesHd5 = {'base' : f}
        
        self.nameMeshPrefix = nameMeshPrefix
        
        
    def registerFile(self,nameFile, label, identifier):
        X, fX = myhd.loadhd5_openFile(nameFile,identifier)
        
        self.filesHd5[identifier + label] = fX
               
        if(identifier == 'Isol' or identifier =='solutions_trans'):
            self.Isol = X
        elif(identifier == 'Wbasis'):
            if(len(label)>0):
                self.WbasisDict[label] = X
            else:
                self.WbasisDict['L2bnd'] = X
                
            
    def closeAllFiles(self):
        for l, f in list(self.filesHd5.items()):
            f.close()
        
    def resetSimulationCase(self, case):
        
        self.getU0 = self.getU0function(case.V0)
        self.getExt = self.getExtensionFunction(case.V0)
        
        self.V0 = case.V0 
        self.ExtOrth = case.ExtOrth 
        
        if(case.V0 == 'P'):
            self.Wbasis = self.WbasisDict[case.RBbasis]
            self.Wbasis2adj = self.WbasisDict[case.RBbasis + '_A']
        else:
            self.Wbasis = self.Wbasis2adj = self.WbasisDict[case.RBbasis]
                  
        if(case.RBbasis == 'L2bnd'):
            self.dotProduct = lambda u,v : assemble(inner(u,v)*self.dsRef)
            self.dotProduct_ = lambda u,v, dm : assemble(inner(u,v)*dm[1])
        elif(case.RBbasis == 'H10'):
            self.dotProduct = lambda u,v : assemble(inner(grad(u),grad(v))*self.dxRef)
            self.dotProduct_ = lambda u,v, dm : assemble(inner(grad(u),grad(v))*dm[0])
        else:
            self.dotProduct = None
            self.dotProduct_  = None
        
    def computeAdjointBasis(self, ns): # for a specific ns
 
        print(".... Now computing adjoint basis for test ", ns)
        
        mesh = fela.EnrichedMesh(self.nameMeshPrefix.format(ns,'xml'))
        V = VectorFunctionSpace(mesh,"CG", 1)
        
        sigmaL0, sigmaEpsL0 = fmts.getSigma_SigmaEps(self.param,mesh,np.zeros((2,2)), op = 'cpp')
        
        for j in range(self.Nmax):
            self.basis.vector().set_local(self.Wbasis2adj[j,:])
            basisAdj = self.getExt(self.basis,V)
            myhd.addDataset(self.filesHd5['base'], basisAdj.vector().get_local(), 'basisAdj/{0}/{1}/{2}'.format(self.V0,ns,j))
            
    def computeU0s(self, ns): # for a specific ns
 
        print(".... Now computing U0 for test ", ns)
        
        mesh = fela.EnrichedMesh(self.nameMeshPrefix.format(ns,'xml'))
        V = VectorFunctionSpace(mesh,"CG", 1)
            
        epsL = self.EpsUnits.reshape((2,2))
        u0 = self.getU0(V, epsL)
        
        myhd.addDataset(self.filesHd5['base'], u0.vector().get_local(), 'U0/{0}/{1}'.format(self.V0,ns))
        
        epsL = self.EpsFluc[ns,:].reshape((2,2))
        u0 = self.getU0(V, epsL)
        
        myhd.addDataset(self.filesHd5['base'], u0.vector().get_local(), 'U0_fluc/{0}/{1}'.format(self.V0,ns))
        
    def computeStressBasis0(self, ns): # for a specific ns
 
        print(".... Now tau_0 for test ", ns)
        
        if(self.V0 == 'T'):
            myhd.addDataset(self.filesHd5['base'], np.zeros(3), 'tau_0_fluc/{0}/{1}'.format(self.V0,ns))
            myhd.addDataset(self.filesHd5['base'], np.zeros(3), 'tau_0/{0}/{1}'.format(self.V0,ns))
        
        else:
            mesh = fela.EnrichedMesh(self.nameMeshPrefix.format(ns,'xml'))
            V = VectorFunctionSpace(mesh,"CG", 2)
            u0 = Function(V)
           
            epsL = self.EpsUnits.reshape((2,2))
            sigmaL, sigmaEpsL = fmts.getSigma_SigmaEps(self.param,mesh,epsL)
            u0.vector().set_local(self.filesHd5['base']['U0/{0}/{1}'.format(self.V0,ns)])
            tau = ten2voigt(fmts.homogenisation(u0, mesh, sigmaL, [0,1], sigmaEpsL))
            myhd.addDataset(self.filesHd5['base'], tau, 'tau_0/{0}/{1}'.format(self.V0,ns))
            
            epsL = self.EpsFluc[ns,:].reshape((2,2))
            sigmaL, sigmaEpsL = fmts.getSigma_SigmaEps(self.param,mesh,epsL)
            u0.vector().set_local(self.filesHd5['base']['U0_fluc/{0}/{1}'.format(self.V0,ns)])
            tau = ten2voigt(fmts.homogenisation(u0, mesh, sigmaL, [0,1], sigmaEpsL))
            myhd.addDataset(self.filesHd5['base'], tau, 'tau_0_fluc/{0}/{1}'.format(self.V0,ns))
        

    def computeStressBasisRB(self, ns): # for a specific ns
 
        print(".... Now tau_0 for test ", ns)
        
        mesh = fela.EnrichedMesh(self.nameMeshPrefix.format(ns,'xml'))
        V = VectorFunctionSpace(mesh,"CG", 2)
        
        if(self.ExtOrth==0): 
        
            VS  = FunctionSpace(mesh, 'DG', 0)
            materials = mesh.subdomains.array().astype('int32')
            materials -= np.min(materials)
            lame = getMyCoeff(materials , self.param, op = 'python') 
            
            lame0 = interpolate(iofe.local_project(lame[0],VS), self.VSref)
            lame1 = interpolate(iofe.local_project(lame[1],VS), self.VSref)
            lame_ = as_vector((lame0,lame1))
            
            sigma = lambda u: fela.sigmaLame(u,lame_)
            
            for j in range(self.Nmax):
                self.basis.vector().set_local(np.array(self.Wbasis[j,:]))
                tau = ten2voigt(feut.Integral(sigma(self.basis),self.dxRef,(2,2)))/self.vol
                myhd.addDataset(self.filesHd5['base'], tau, 'tau/{0}/{1}/{2}/{3}'.format(self.V0,self.ExtOrth,ns,j))
            
        else:
            sigmaL0, sigmaEpsL0 = fmts.getSigma_SigmaEps(self.param,mesh,np.zeros((2,2)))
            
            basisExt = Function(V)
            
            for j in range(self.Nmax):  
                basisExt.vector().set_local(self.filesHd5['base']['basisAdj/{0}/{1}/{2}'.format(self.V0,ns,j)])
                tau = ten2voigt(fmts.homogenisation(basisExt, mesh, sigmaL0, [0,1], sigmaEpsL0))
                myhd.addDataset(self.filesHd5['base'], tau, 'tau/{0}/{1}/{2}/{3}'.format(self.V0,self.ExtOrth,ns,j))
        
    def computeEtas(self, ns): # for a specific ns
        
        print(".... Now etas for test ", ns)
       
        mesh = fela.EnrichedMesh(self.nameMeshPrefix.format(ns,'xml'))
        V = VectorFunctionSpace(mesh,"CG", 1)
        
        u0 = Function(V)         
        print('U0/{0}/{1}'.format(self.V0,ns))
        print(self.filesHd5['base'].keys())
        u0.vector().set_local(self.filesHd5['base']['U0/{0}/{1}'.format(self.V0,ns)])
        
        self.urefAux.vector().set_local(np.array(self.Isol[ns,:]))
        
        etas = np.zeros(self.Nmax)
        if(self.ExtOrth==0): 
        
            self.urefAux2.interpolate(u0)
            for j in range(self.Nmax):
                self.basis.vector().set_local(np.array(self.Wbasis[j,:]))
                etas[j] = self.dotProduct(self.basis, self.urefAux - self.urefAux2)
                
        else:
            dm = [Measure('dx', mesh), Measure('ds', mesh)]
            basisExt = Function(V)
            uVaux = interpolate(self.urefAux,V)
            for j in range(self.Nmax):
                basisExt.vector().set_local(self.filesHd5['base']['basisAdj/{0}/{1}/{2}'.format(self.V0,ns,j)])
                etas[j] = self.dotProduct_(basisExt, uVaux - u0, dm)
                
        myhd.addDataset(self.filesHd5['base'], etas, 'etas/{0}/{1}/{2}'.format(self.V0,self.ExtOrth,ns))
        
        
    def getU0function(self, V0):
        if(V0 == 'T'):
            return lambda W, epsbar : Function(W)
        elif(V0 == 'L'):
            return lambda W, epsbar : mpms.solveMultiscale(self.param[0:2,:], W.mesh(), epsbar, op = 'Lin', others = {'polyorder': 2, 'bdr' : 2})[0]
        elif(V0 == 'P'):
            return lambda W, epsbar : mpms.solveMultiscale(self.param[0:2,:], W.mesh(), epsbar, op = 'periodic', others = [1./3.,2./3.,1./3.,2./3.])[0]
        elif(V0 == 'M'):
            return lambda W, epsbar : mpms.solveMultiscale(self.param[0:2,:], W.mesh(), epsbar, op = 'MR')[0]
        else:
            print('Wrong option ', V0)
            input()
        
               
    def getExtensionFunction(self, V0):
        if(V0 == 'T'):
            return lambda w,W : interpolate(w,W)
        elif(V0 == 'L'):
            return lambda w,W : mpms.solveMultiscale(self.param[0:2,:], W.mesh(), np.zeros((2,2)), op = 'BCdirich_lag', others = {'polyorder': 2, 'uD' : w})[0]
        elif(V0 == 'P'):
            return lambda w,W : mpms.solveMultiscale(self.param[0:2,:], W.mesh(), np.zeros((2,2)), op = 'BCdirich_lag', others = [w])[0] # but should be antiperiodic
        elif(V0 == 'M'):
            return lambda w,W : interpolate(w,W)
        else:
            print('Wrong option ', V0)
            input()
        
             
    

    
    # def computeAdjointBasis(self, ns): # for a specific ns
 
        
    #     for i in range(ns):
    #     print(".... Now computing tau for test ", i)
        
    #     mesh = fela.EnrichedMesh(nameMeshPrefix.format(i,'xml'))
    #     V = VectorFunctionSpace(mesh,"CG", 1)
    #     VS  = FunctionSpace(mesh, 'DG', 0)
        
    #     materials = mesh.subdomains.array().astype('int32')
    #     materials -= np.min(materials)
    #     lame = getMyCoeff(materials , param, op = 'python') 
        
    #     lame0 = interpolate(iofe.local_project(lame[0],VS), VSref)
    #     lame1 = interpolate(iofe.local_project(lame[1],VS), VSref)
    #     lame_ = as_vector((lame0,lame1))
        
    #     sigma = lambda u: fela.sigmaLame(u,lame_)        
               
    #     epsL = EpsUnits.reshape((2,2))
    #     u0 = getu0(V, epsL)
    #     sigmaL, sigmaEpsL = fmts.getSigma_SigmaEps(param,mesh,epsL)
    #     tau[1][i,:] = ten2voigt(fmts.homogenisation( u0, mesh, sigmaL, [0,1], sigmaEpsL))
        
    #     epsL = EpsFluc[i,:].reshape((2,2))
    #     u0 = getu0(V, epsL)
    #     sigmaL, sigmaEpsL = fmts.getSigma_SigmaEps(param,mesh,epsL)
    #     tau[2][i,:] = ten2voigt(fmts.homogenisation( u0, mesh, sigmaL, [0,1], sigmaEpsL))
        
    #     sigmaL0, sigmaEpsL0 = fmts.getSigma_SigmaEps(param,mesh,np.zeros((2,2)))
                
    #     for j in range(Nmax):
    #         if(Orth):
    #             basisAux.vector().set_local(WbasisAux[j,:])
    #             Ibasis = getOrth(basisAux,V)
    #             tau[0][i,j,:] = ten2voigt(fmts.homogenisation(Ibasis, mesh, sigmaL0, [0,1], sigmaEpsL0))

    #         else:    
    #             basis.vector().set_local(np.array(Wbasis[j,:]))
    #             tau[0][i,j,:] = ten2voigt(feut.Integral(sigma(basis),dxRef,(2,2)))/vol
        
    
def getStressBasis_noMesh(tau, Wbasis, Isol, ellipseData, Nmax, Vref, param, EpsDirection): # tau = [ tau_basis, tau_0, tau_0_fluc]
    
    ns = ellipseData.shape[0]
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
    
    Mref = Vref.mesh()
    normal = FacetNormal(Mref)
    dsRef = Measure('ds', Mref)

    
    EpsUnits = np.array([[1.,0.,0.,0.],[0.,0.5,0.5,0.]])[EpsDirection,:]
    
    basis = Function(Vref)
    usol = Function(Vref)
    
    others = {'polyorder' : 2, 'bdr' : 2, 'uD' : basis}
    
    transformBasis0 = lambda mesh, epsbar : mpms.solveMultiscale(param[0:2,:], mesh, epsbar, op = 'Lin', others = others)[0]
    transformBasis = lambda mesh : mpms.solveMultiscale(param[0:2,:], mesh, np.zeros((2,2)), op = 'BCdirich_lag', others = others)[0]
    
    for i in range(ns):
        print(".... Now computing tau for test ", i)

        meshGMSH = meut.ellipseMesh2(ellipseData[i,:4,:], x0 = -1.0, y0 = -1.0, Lx = 2.0 , Ly = 2.0 , lcar = 0.1)
        meshGMSH.setTransfiniteBoundary(21)
        meshGMSH.setNameMesh("mesh_reduced_temp.xdmf")
        mesh = meshGMSH.getEnrichedMesh() 

        V = VectorFunctionSpace(mesh,"CG", 2)
        
        basis.vector().set_local(np.zeros(Vref.dim()))
        usol.vector().set_local(Isol[i,:])
                
        epsL = EpsUnits.reshape((2,2)) + feut.Integral(outer(usol,normal), dsRef, shape = (2,2))/4.0
        Ibasis = transformBasis0(mesh,epsL)
        sigmaL, sigmaEpsL = fmts.getSigma_SigmaEps(param,mesh,epsL, op = 'cpp')
        tau[1][i,:] = ten2voigt(fmts.homogenisation(Ibasis, mesh, sigmaL, [0,1], sigmaEpsL))
                
        sigmaL0, sigmaEpsL0 = fmts.getSigma_SigmaEps(param,mesh,np.zeros((2,2)), op = 'cpp')
        
        for j in range(Nmax):
            basis.vector().set_local(Wbasis[j,:])
            # a = np.zeros(2)
            # B = -feut.Integral(outer(basis,normal), dsRef, shape = (2,2))/4.0
            # T = feut.affineTransformationExpession(a,B,Mref)
            # others['uD'] = basis + T
            Ibasis = transformBasis(mesh)
            tau[0][i,j,:] = ten2voigt(fmts.homogenisation(Ibasis, mesh, sigmaL0, [0,1], sigmaEpsL0))




            