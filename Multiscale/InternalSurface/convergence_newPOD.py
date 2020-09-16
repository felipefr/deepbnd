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

normL2 = lambda x,dx : np.sqrt(assemble(inner(x,x)*dx))


folder = "./debugData/"
radFile = folder + "RVE_POD_{0}_.{1}"

BC = 'big_periodic'
BCtest = 'test_periodic'

# Nbasis = np.array([ 3 + int(i**2.04) for i in range(1,12)] + [155,160])
# Nbasis = np.array([ 3 + int(i**2.04) for i in range(1,12)] + [155,160])
Nbasis = np.array([2,8,15,30,60,100,130,150,156])
Nmax = np.max(Nbasis)
    
S = np.loadtxt(folder + 'snapshots_{0}.txt'.format(BC))
Stest = np.loadtxt(folder + 'snapshots_{0}.txt'.format(BCtest))

EpsFluc = np.loadtxt(folder + 'Eps_{0}.txt'.format(BCtest))
StressList = np.loadtxt(folder + 'Stress_{0}.txt'.format(BCtest))

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

mu1 = elut.eng2mu(nu1,E1)
lamb1 = elut.eng2lambPlane(nu1,E1)
mu2 = elut.eng2mu(nu2,E2)
lamb2 = elut.eng2lambPlane(nu2,E2)

eps0 = np.zeros((2,2))
eps0[0,0] = 0.1
eps0 = 0.5*(eps0 + eps0.T)

param = np.array([[lamb1, mu1], [lamb2,mu2]])    

g = gmts.displacementGeneratorBoundary(x0L,y0L, LxL, LyL, NpLxL)

Ntest = 25
NNbasis = len(Nbasis) 
    
normUN_Uex_L2 = np.zeros((Ntest,NNbasis)); normUN_Uex_L2_rel = np.zeros((Ntest,NNbasis))
normUR_Uex_L2 = np.zeros((Ntest,NNbasis)); normUR_Uex_L2_rel = np.zeros((Ntest,NNbasis)) 
normUN_Uex_L2domain = np.zeros((Ntest,NNbasis)); normUN_Uex_L2domain_rel = np.zeros((Ntest,NNbasis))
normUN_Uex_Rm = np.zeros((Ntest,NNbasis)); normUN_Uex_Rm_rel = np.zeros((Ntest,NNbasis))
normUR_Uex_Rm = np.zeros((Ntest,NNbasis)); normUR_Uex_Rm_rel = np.zeros((Ntest,NNbasis)) 
normUR_Uex_L2domain = np.zeros((Ntest,NNbasis)); normUR_Uex_L2domain_rel = np.zeros((Ntest,NNbasis))

sigma_11_N = np.zeros((Ntest,NNbasis)); 
sigma_error_N = np.zeros((Ntest,NNbasis)); sigma_error_N_rel = np.zeros((Ntest,NNbasis))  
sigma_11_R = np.zeros((Ntest,NNbasis)); 
sigma_error_R = np.zeros((Ntest,NNbasis)); sigma_error_R_rel = np.zeros((Ntest,NNbasis))

normUex_L2domain = np.zeros(Ntest)
normUex_L2 = np.zeros(Ntest) 
normUex_Rm = np.zeros(Ntest) 
sigma_ref = np.zeros((Ntest,4))

Wbasis, sig, Mhsqrt, Mh = fmts.pod_customised(S, 'L2', [g])
Wbasis_, sig_, Mhsqrt_, Mh_ = fmts.pod_customised(S, 'l2', [g])

bbasis = []
for j in range(Nmax):
    bbasis.append(fmts.PointExpression(Wbasis[:,j], g))

for nt in range(Ntest):
    print(".... Now computing test ", nt)
    # Generating reference solution
    ellipseData = np.loadtxt(folder + 'ellipseData_' + str(nt) + '.txt')[:NL]
    
    meshGMSH = gmsh.ellipseMesh2(ellipseData, x0L, y0L, LxL, LyL, lcar, ifPeriodic)
    meshGMSH.setTransfiniteBoundary(NpLxL)
    
    mesh = feut.getMesh(meshGMSH, '_reference', radFile)
    
    BM = fmts.getBMrestriction(g, mesh)        
    UL_ref_ = Stest[:,nt]
    
    UL_ref = fmts.PointExpression(UL_ref_, g, 'python') 
    errorMR = BM.T @ UL_ref_
    print('errorMR' , errorMR)
    
    epsL = eps0 + EpsFluc[nt,:].reshape((2,2))
    
    Uref = mpms.solveMultiscale(param, mesh, epsL, op = 'BCdirich_lag', others = [UL_ref])[0]
    
    sigmaL, sigmaEpsL = fmts.getSigma_SigmaEps(param,mesh,epsL)
     
    sigma_ref[nt,:] = fmts.homogenisation(Uref, mesh, sigmaL, [0,1], sigmaEpsL).flatten()
    
    print("Checking homogenisation : ", sigma_ref[nt,:], StressList[nt,:])
    
    # end mesh creation
    
    alpha = np.zeros(Nmax)
    
    normUex_L2domain[nt] = normL2(Uref,mesh.dx) 
    normUex_L2[nt] = normL2(Uref,mesh.ds) 
    normUex_Rm[nt] = np.linalg.norm(UL_ref_) 

    meas_ds = assemble( Constant(1.0)*mesh.ds )    
    for j in range(Nmax):
        alpha[j] = assemble( inner(bbasis[j], Uref)*mesh.ds )/meas_ds
    
    
    for i,N in enumerate(Nbasis): 
        UN = mpms.solveMultiscale(param, mesh, epsL, op = 'POD', others = [bbasis[0:N], alpha[0:N]])[0]
        
        ULR_vec_ = Wbasis_[:,:N] @ Wbasis_[:,:N].T @ UL_ref_
        ULR = fmts.PointExpression(ULR_vec_ ,g)
        UR = mpms.solveMultiscale(param, mesh, epsL, op = 'BCdirich_lag', others = [ULR])[0]
        
        sigma_i_N = fmts.homogenisation(UN, mesh, sigmaL, [0,1], sigmaEpsL).flatten()
        sigma_i_R = fmts.homogenisation(UR, mesh, sigmaL, [0,1], sigmaEpsL).flatten()
        
        sigma_11_R[nt,i] = sigma_i_R[0]
        sigma_11_N[nt,i] = sigma_i_N[0]
        
        sigma_error_R[nt,i] = np.abs(sigma_11_R[nt,i] - sigma_ref[nt,0])
        sigma_error_N[nt,i] = np.abs(sigma_11_N[nt,i] - sigma_ref[nt,0])
        
        normUN_Uex_L2[nt,i] = normL2(UN - Uref, mesh.ds)
        normUR_Uex_L2[nt,i] = normL2(UR - Uref, mesh.ds)
        normUN_Uex_L2domain[nt,i] = normL2(UN - Uref, mesh.dx)   
        normUR_Uex_L2domain[nt,i] = normL2(UR - Uref, mesh.dx)
        
        normUN_Uex_Rm[nt,i] = np.linalg.norm( g(mesh,UN).flatten()  - UL_ref_) 
        normUR_Uex_Rm[nt,i] = np.linalg.norm( ULR_vec_ - UL_ref_)
        
    sigma_error_N_rel[nt,:] = sigma_error_N[nt,:]/sigma_ref[nt,0]
    sigma_error_R_rel[nt,:] = sigma_error_R[nt,:]/sigma_ref[nt,0]
    normUN_Uex_L2_rel[nt,:] = normUN_Uex_L2[nt,:]/normUex_L2[nt]
    normUR_Uex_L2_rel[nt,:] = normUR_Uex_L2[nt,:]/normUex_L2[nt]
    normUN_Uex_L2domain_rel[nt,:] = normUN_Uex_L2domain[nt,:]/normUex_L2domain[nt]
    normUR_Uex_L2domain_rel[nt,:] = normUR_Uex_L2domain[nt,:]/normUex_L2domain[nt]
    normUN_Uex_Rm_rel[nt,:] = normUN_Uex_Rm[nt,:]/normUex_Rm[nt]
    normUR_Uex_Rm_rel[nt,:] = normUR_Uex_Rm[nt,:]/normUex_Rm[nt]

cumsum = np.cumsum(sig*sig)
expectedError = np.sqrt(1-(cumsum[:-1]/cumsum[-1]))

plt.figure(1,(10,8))
plt.subplot('121')
plt.title('Absolute error fluc L2')
plt.plot(Nbasis, np.mean(normUN_Uex_L2, axis = 0),'-ro', label = 'normUN_Uex_L2')
plt.plot(Nbasis, np.mean(normUN_Uex_L2, axis = 0) + np.std(normUN_Uex_L2, axis = 0) ,'r--')
plt.plot(Nbasis, np.mean(normUN_Uex_L2, axis = 0) - np.std(normUN_Uex_L2, axis = 0) ,'r--')
plt.plot(Nbasis, np.mean(normUR_Uex_L2, axis = 0),'-bo', label = 'normUR_Uex_L2')
plt.plot(Nbasis, np.mean(normUR_Uex_L2, axis = 0) + np.std(normUR_Uex_L2, axis = 0) ,'b--')
plt.plot(Nbasis, np.mean(normUR_Uex_L2, axis = 0) - np.std(normUR_Uex_L2, axis = 0) ,'b--')

plt.plot(Nbasis, np.mean(normUN_Uex_L2domain, axis = 0),'-go', label = 'normUN_Uex_L2domain')
plt.plot(Nbasis, np.mean(normUN_Uex_L2domain, axis = 0) + np.std(normUN_Uex_L2domain, axis = 0) ,'g--')
plt.plot(Nbasis, np.mean(normUN_Uex_L2domain, axis = 0) - np.std(normUN_Uex_L2domain, axis = 0) ,'g--')
plt.plot(Nbasis, np.mean(normUR_Uex_L2domain, axis = 0),'-co', label = 'normUR_Uex_L2domain')
plt.plot(Nbasis, np.mean(normUR_Uex_L2domain, axis = 0) + np.std(normUR_Uex_L2domain, axis = 0) ,'c--')
plt.plot(Nbasis, np.mean(normUR_Uex_L2domain, axis = 0) - np.std(normUR_Uex_L2domain, axis = 0) ,'c--')

plt.xlabel('N')
plt.ylabel('error fluc')
plt.yscale('log')
plt.grid()
plt.legend()

plt.subplot('122')
plt.title('Relative error fluc L2')
plt.plot(Nbasis, np.mean(normUN_Uex_L2_rel, axis = 0),'-ro', label = 'normUN_Uex_L2')
plt.plot(Nbasis, np.mean(normUN_Uex_L2_rel, axis = 0) + np.std(normUN_Uex_L2_rel, axis = 0) ,'r--')
plt.plot(Nbasis, np.mean(normUN_Uex_L2_rel, axis = 0) - np.std(normUN_Uex_L2_rel, axis = 0) ,'r--')
plt.plot(Nbasis, np.mean(normUR_Uex_L2_rel, axis = 0),'-bo', label = 'normUR_Uex_L2')
plt.plot(Nbasis, np.mean(normUR_Uex_L2_rel, axis = 0) + np.std(normUR_Uex_L2_rel, axis = 0) ,'b--')
plt.plot(Nbasis, np.mean(normUR_Uex_L2_rel, axis = 0) - np.std(normUR_Uex_L2_rel, axis = 0) ,'b--')

plt.plot(Nbasis, np.mean(normUN_Uex_L2domain_rel, axis = 0),'-go', label = 'normUN_Uex_L2domain')
plt.plot(Nbasis, np.mean(normUN_Uex_L2domain_rel, axis = 0) + np.std(normUN_Uex_L2domain_rel, axis = 0) ,'g--')
plt.plot(Nbasis, np.mean(normUN_Uex_L2domain_rel, axis = 0) - np.std(normUN_Uex_L2domain_rel, axis = 0) ,'g--')
plt.plot(Nbasis, np.mean(normUR_Uex_L2domain_rel, axis = 0),'-co', label = 'normUR_Uex_L2domain')
plt.plot(Nbasis, np.mean(normUR_Uex_L2domain_rel, axis = 0) + np.std(normUR_Uex_L2domain_rel, axis = 0) ,'c--')
plt.plot(Nbasis, np.mean(normUR_Uex_L2domain_rel, axis = 0) - np.std(normUR_Uex_L2domain_rel, axis = 0) ,'c--')

plt.plot(np.arange(1,160), expectedError, 'k--', label = 'expected')
plt.xlabel('N')
plt.ylabel('rel error fluc')
plt.yscale('log')
plt.grid()
plt.legend()

plt.savefig('errorDisp_batch.png')

plt.figure(3,(10,5))
plt.subplot('121')
plt.title('stress')
plt.plot(Nbasis, np.mean(sigma_11_N, axis = 0),'-ro', label = 'stress UN')
plt.plot(Nbasis, np.mean(sigma_11_N, axis = 0) + np.std(sigma_11_N, axis = 0), 'r--')
plt.plot(Nbasis, np.mean(sigma_11_N, axis = 0) - np.std(sigma_11_N, axis = 0),'r--')
plt.plot(Nbasis, np.mean(sigma_11_R, axis = 0),'-bo', label = 'stress UR')
plt.plot(Nbasis, np.mean(sigma_11_R, axis = 0) + np.std(sigma_11_R, axis = 0), 'b--')
plt.plot(Nbasis, np.mean(sigma_11_R, axis = 0) - np.std(sigma_11_R, axis = 0),'b--')
plt.xlabel('N')
plt.ylabel('sigma_11')
plt.plot([Nbasis[0],Nbasis[-1]], 2*[np.mean(sigma_ref[:,0])],'k', label = 'stress reference')
plt.plot([Nbasis[0],Nbasis[-1]], 2*[np.mean(sigma_ref[:,0]) + np.std(sigma_ref[:,0])],'k--')
plt.plot([Nbasis[0],Nbasis[-1]], 2*[np.mean(sigma_ref[:,0]) - np.std(sigma_ref[:,0])],'k--')

plt.legend()
plt.grid()

plt.subplot('122')
plt.title('relative error stress')
plt.plot(Nbasis, np.mean(sigma_error_N_rel, axis = 0),'-ro', label = 'stress UN')
plt.plot(Nbasis, np.mean(sigma_error_N_rel, axis = 0) + np.std(sigma_error_N_rel, axis = 0), 'r--')
plt.plot(Nbasis, np.mean(sigma_error_N_rel, axis = 0) - np.std(sigma_error_N_rel, axis = 0),'r--')
plt.plot(Nbasis, np.mean(sigma_error_R_rel, axis = 0),'-bo', label = 'stress UR')
plt.plot(Nbasis, np.mean(sigma_error_R_rel, axis = 0) + np.std(sigma_error_R_rel, axis = 0), 'b--')
plt.plot(Nbasis, np.mean(sigma_error_R_rel, axis = 0) - np.std(sigma_error_R_rel, axis = 0),'b--')

plt.xlabel('N')
plt.ylabel('|sigma_11(N) - sigma_11_ref|/|sigma_11_ref|' )
plt.yscale('log')
plt.grid()
plt.legend()

plt.savefig('stressError_batch.png')


plt.show()

