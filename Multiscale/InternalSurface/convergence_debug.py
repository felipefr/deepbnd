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

folder = "./debugData/"
radFile = folder + "RVE_POD_{0}_.{1}"

BC = 'MR'
idEllipse = 10
Nbasis = np.array([2,4,8,12,18,25,35,50,70,85,100,120,145,156])
# Nbasis = np.array([4,10,50,100,130,160])

S = np.loadtxt(folder + 'snapshots_{0}.txt'.format(BC))
# S = np.loadtxt('./dataDebug/snapshots_MR.txt')

EpsFluc = np.loadtxt(folder + 'Eps{0}.txt'.format(BC))
StressList = np.loadtxt(folder + 'Stress{0}.txt'.format(BC))

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

# Generating reference solution
ellipseData = np.loadtxt(folder + 'ellipseData_' + str(idEllipse) + '.txt')[:NL]

meshGMSH = gmsh.ellipseMesh2(ellipseData, x0L, y0L, LxL, LyL, lcar, ifPeriodic)
meshGMSH.setTransfiniteBoundary(NpLxL)

mesh = feut.getMesh(meshGMSH, '_reference', radFile)

BM = fmts.getBMrestriction(g, mesh)        
UL_ref_ = S[:,idEllipse]

UL_ref = fmts.PointExpression(UL_ref_, g, 'python') 
errorMR = BM.T @ UL_ref_
print('errorMR' , errorMR)

epsL = eps0 + EpsFluc[idEllipse,:].reshape((2,2))

Uref = mpms.solveMultiscale(param, mesh, epsL, op = 'BCdirich_lag', others = [UL_ref])[0]

sigmaL, sigmaEpsL = fmts.getSigma_SigmaEps(param,mesh,epsL)
 
sigma_ref = fmts.homogenisation(Uref, mesh, sigmaL, [0,1], sigmaEpsL).flatten()

print(sigma_ref, StressList[idEllipse,:])

# end mesh creation

Nmax = np.max(Nbasis)
alpha = np.zeros(Nmax)
alpha2 = np.zeros(Nmax)

Wbasis,sig , Mhsqrt, Mh= fmts.pod_customised(S, 'L2', [g,mesh]) 
# WbasisBar, Wbasis, sig, Mhsqrt, Mh, MhsqrtInv = fmts.pod_customised_2(S, 'L2', [g,mesh])
# Wbasis,sig, Mhsqrt, Mh = fmts.pod_customised(S, Nmax, 'l2') 

bbasis = []

start = timer()
meas_ds = assemble( Constant(1.0)*mesh.ds )

for j in range(Nmax):
    bbasis.append(fmts.PointExpression(Wbasis[:,j], g))
    alpha[j] = assemble( inner(bbasis[-1], Uref)*mesh.ds )/meas_ds
    # alpha[j] = np.dot(Mh@ Wbasis[:,j], UL_ref_)/meas_ds

end = timer()
print('computing alpha', end - start) # Time in seconds, e.g. 5.38091952400282

UN = []


for i,N in enumerate(Nbasis): 
    UN.append(mpms.solveMultiscale(param, mesh, epsL, op = 'POD', others = [bbasis[0:N], alpha[0:N]])[0])
    # UN.append(mpms.solveMultiscale(param, mesh, epsL, op = 'POD_noMR', others = [bbasis[0:N], alpha[0:N]])[0])
    
normUN_Uex_L2 = [] ; normUR_Uex_L2 = [] ; normUN_UR_L2 = []  ; normUN_Uex_L2domain = []
normUN_Uex_Rm = [] ; normUR_Uex_Rm = [] ; normUN_UR_Rm = []  ; normUR_Uex_L2domain = []

sigma_11 = []
sigma_error = []  
sigma_11_strong = []
sigma_error_strong = []  

normL2 = lambda x,dx : np.sqrt(assemble(inner(x,x)*dx))

# normUex_L2domain = np.sqrt( normL2(UrefComplete[0],mesh2.dx(0))**2.0 + normL2(UrefComplete[0],mesh2.dx(1))**2.0 )
normUex_L2domain = normL2(Uref,mesh.dx) 
normUex_L2 = normL2(Uref,mesh.ds) 
normUex_Rm = np.linalg.norm(UL_ref_) 

UNstrong = []

for i,N in enumerate(Nbasis):
    # UR_vec_ = Mhsqrt @ WbasisBar[:,:N] @ WbasisBar[:,:N].T @  Mhsqrt @ UL_ref_
    UR_vec_ = Mhsqrt @ Wbasis[:,:N] @ Wbasis[:,:N].T @  Mhsqrt @ UL_ref_
    # UR_vec_ = Wbasis[:,:N] @ Wbasis[:,:N].T @ UL_ref_
    
    UR = fmts.PointExpression(UR_vec_ ,g)
    
    UNstrong.append(mpms.solveMultiscale(param, mesh, epsL, op = 'BCdirich_lag', others = [UR])[0])
        
    normal = FacetNormal(mesh)
    print(feut.Integral(UN[i],mesh.dx,shape = (2,)))
    print(feut.Integral(outer(UN[i],normal),mesh.ds ,shape = (2,2)))
    print(feut.Integral(UNstrong[-1],mesh.dx,shape = (2,)))
    print(feut.Integral(outer(UNstrong[-1],normal),mesh.ds ,shape = (2,2)))

    # print(Wbasis[:,0:4].T @ Mh @ g(mesh,UN[i][0]).flatten())
    
    # print(np.sqrt(assemble(dot(Ulist[i][0],Ulist[i][0])*mesh.dx)))
    
    sigma_i_strong = fmts.homogenisation(UNstrong[-1], mesh, sigmaL, [0,1], sigmaEpsL).flatten()
    
    sigma_i = fmts.homogenisation(UN[i], mesh, sigmaL, [0,1], sigmaEpsL).flatten()
    
    sigma_11_strong.append(sigma_i_strong[0])
    
    sigma_11.append(sigma_i[0])
    
    sigma_error.append(np.abs(sigma_i[0] - sigma_ref[0]))

    sigma_error_strong.append(np.abs(sigma_i_strong[0] - sigma_ref[0]))
    
    # plt.figure(i,(10,18))
    # plt.subplot('321')
    # plot(Uref[0])
    # plt.xlim(0.33333,0.666666)
    # plt.ylim(0.33333,0.666666)
    # plt.subplot('322')
    # plot(Uref[1])
    # plt.xlim(0.33333,0.666666)
    # plt.ylim(0.33333,0.666666)
    # plt.subplot('323')
    # plot(UN[i][0])
    # plt.subplot('324')
    # plot(UN[i][1])
    # plt.subplot('325')
    # plot(UNstrong[-1][0])
    # plt.subplot('326')
    # plot(UNstrong[-1][1])

    normUN_Uex_L2.append(normL2(UN[i] - Uref, mesh.ds))
    normUR_Uex_L2.append(normL2(UR - Uref, mesh.ds))
    normUN_UR_L2.append(normL2(UN[i] - UR, mesh.ds))
    normUN_Uex_L2domain.append(normL2(UN[i] - Uref, mesh.dx))   
    normUR_Uex_L2domain.append(normL2(UNstrong[-1] - Uref, mesh.dx))

    normUN_Uex_Rm.append( np.linalg.norm( g(mesh,UN[i]).flatten()  - UL_ref_)) 
    normUR_Uex_Rm.append(np.linalg.norm( UR_vec_ - UL_ref_))
    normUN_UR_Rm.append(np.linalg.norm( g(mesh,UN[i]).flatten()  - UR_vec_ ))
    
sigma_error_rel = np.array(sigma_error)/sigma_ref[0]
sigma_error_strong_rel = np.array(sigma_error_strong)/sigma_ref[0]
normUN_Uex_L2_rel = np.array(normUN_Uex_L2)/normUex_L2
normUR_Uex_L2_rel = np.array(normUR_Uex_L2)/normUex_L2
normUN_UR_L2_rel = np.array(normUN_UR_L2)/normUex_L2
normUN_Uex_L2domain_rel = np.array(normUN_Uex_L2domain)/normUex_L2domain
normUR_Uex_L2domain_rel = np.array(normUR_Uex_L2domain)/normUex_L2domain
normUN_Uex_Rm_rel = np.array(normUN_Uex_Rm)/normUex_Rm
normUR_Uex_Rm_rel = np.array(normUR_Uex_Rm)/normUex_Rm
normUN_UR_Rm_rel = np.array(normUN_UR_Rm)/normUex_Rm

cumsum = np.cumsum(sig*sig)
expectedError = np.sqrt(1-(cumsum[:-1]/cumsum[-1]))

plt.figure(1,(10,8))
plt.subplot('121')
plt.title('Absolute error fluc L2')
plt.plot(Nbasis, normUN_Uex_L2,'-o', label = 'normUN_Uex_L2')
plt.plot(Nbasis, normUR_Uex_L2,'-o', label = 'normUR_Uex_L2')
# plt.plot(Nbasis, normUN_UR_L2, '-o', label = 'normUN_UR_L2')
plt.plot(Nbasis, normUN_Uex_L2domain, '-o', label = 'normUN_Uex_L2domain')
plt.plot(Nbasis, normUR_Uex_L2domain, '-o', label = 'normUR_Uex_L2domain')
plt.xlabel('N')
plt.ylabel('error fluc')
plt.yscale('log')
plt.grid()
plt.legend()

plt.subplot('122')
plt.title('Relative error fluc L2')
plt.plot(Nbasis, normUN_Uex_L2_rel,'-o', label = 'normUN_Uex_L2')
plt.plot(Nbasis, normUR_Uex_L2_rel,'-o', label = 'normUR_Uex_L2')
plt.plot(Nbasis, normUN_UR_L2_rel, '-o', label = 'normUN_UR_L2')
plt.plot(Nbasis, normUN_Uex_L2domain_rel, '-o', label = 'normUN_Uex_L2domain')
plt.plot(Nbasis, normUR_Uex_L2domain_rel, '-o', label = 'normURe_Uex_L2domain')
plt.plot(np.arange(1,160), expectedError, '--', label = 'expected')
# plt.plot(np.arange(1,160), expectedError_, '--', label = 'expected_')
plt.xlabel('N')
plt.ylabel('rel error fluc')
plt.yscale('log')
plt.grid()
plt.legend()

plt.savefig('errorDisp_case7_0.png')

plt.figure(2,(10,6))
plt.subplot('121')
plt.title('stress')
plt.plot(Nbasis, sigma_11,'-o', label = 'stress UN')
plt.plot(Nbasis, sigma_11_strong,'-o', label = 'stress UR')
plt.xlabel('N')
plt.ylabel('sigma_11')
plt.plot([Nbasis[0],Nbasis[-1]], 2*[sigma_ref[0]],'-o', label = 'stress reference')
plt.legend()
plt.grid()

plt.subplot('122')
plt.title('relative error stress')
plt.plot(Nbasis, sigma_error_rel,'-o', label = 'stress UN')
plt.plot(Nbasis, sigma_error_strong_rel,'-o', label = 'stress UR')
plt.xlabel('N')
plt.ylabel('|sigma_11(N) - sigma_11_ref|/|sigma_11_ref|' )
plt.yscale('log')
plt.grid()
plt.legend()

plt.savefig('stressError_case7_0.png')


plt.show()
