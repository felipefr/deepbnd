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

S = np.loadtxt('snapshots_periodic_balanced.txt')[:,:200]

# Wbasis, sig, VT = np.linalg.svd(S)

folder = "./data1/"
radFile = folder + "RVE_POD_ref_{0}.{1}"

folder2 = "./dataConvergence/"
radFile2 = folder2 + "RVE_POD_ref_{0}.{1}"


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

idEllipse = 0

ellipseData = np.loadtxt(folder + 'ellipseData_' + str(idEllipse) + '.txt')[:NL,:]

meshGMSH = gmsh.ellipseMesh2(ellipseData, x0L, y0L, LxL, LyL , lcar, ifPeriodic)
meshGMSH.setTransfiniteBoundary(NpLxL)

print("nodes per side",  int(Lx/lcar) + 1)

meshGeoFile = radFile2.format(times,'geo')
meshXmlFile = radFile2.format(times,'xml')
meshMshFile = radFile2.format(times,'msh')

meshGMSH.write(meshGeoFile,'geo')
os.system('gmsh -2 -algo del2d -format msh2 ' + meshGeoFile)

os.system('dolfin-convert {0} {1}'.format(meshMshFile, meshXmlFile))

mesh = fela.EnrichedMesh(meshXmlFile)

# end mesh creation

contrast = 10.0
E2 = 1.0
E1 = contrast*E2 # inclusions
nu1 = 0.3
nu2 = 0.3

mu1 = elut.eng2mu(nu1,E1)
lamb1 = elut.eng2lambPlane(nu1,E1)
mu2 = elut.eng2mu(nu2,E2)
lamb2 = elut.eng2lambPlane(nu2,E2)

eps = np.zeros((2,2))
eps[0,0] = 0.1
eps = 0.5*(eps + eps.T)

param = np.array([[lamb1, mu1], [lamb2,mu2]])    

Nbasis = np.array([ 3 + int(i**2.04) for i in range(1,12)])
# Nbasis = np.array([5,10,15])

sigma, sigmaEps = fmts.getSigma_SigmaEps(param,mesh,eps)
g = gmts.displacementGeneratorBoundary(x0L,y0L, LxL, LyL, NpLxL)
UrefB_dof = S[:,idEllipse]
UrefB = fmts.PointExpression(UrefB_dof, g)

Nmax = np.max(Nbasis)
alpha = np.zeros(Nmax)

Wbasis,sig , Mhsqrt, Mh= fmts.pod_customised(S, Nmax, 'L2', [g,mesh]) 

bbasis = []

start = timer()
for j in range(Nmax):
    bbasis.append(fmts.PointExpression(Wbasis[:,j], g))
    alpha[j] = 0.25*assemble( inner(bbasis[-1],UrefB)*mesh.ds )

end = timer()
print('computing alpha', end - start) # Time in seconds, e.g. 5.38091952400282

UN = []


for i,N in enumerate(Nbasis): 
    UN.append(mpms.solveMultiscale(param, mesh, eps, op = 'POD', others = [bbasis[0:N], alpha[0:N]]))
    
normUN_Uex_L2 = [] ; normUR_Uex_L2 = [] ; normUN_UR_L2 = []  ; normUN_Uex_L2domain = []
normUN_Uex_Rm = [] ; normUR_Uex_Rm = [] ; normUN_UR_Rm = [] 

sigma_11 = []
# sigma_error = []  
# sigma_error_rel = [] 

normL2 = lambda x,dx : np.sqrt(assemble(inner(x,x)*dx))

# normUex_L2domain = normL2(Uref[0],mesh.dx) 
normUex_L2 = normL2(UrefB,mesh.ds) 
normUex_Rm = np.linalg.norm(UrefB_dof) 

for i,N in enumerate(Nbasis):
    UR_vec = Mhsqrt @ Wbasis[:,:N] @ Wbasis[:,:N].T @  Mhsqrt @ UrefB_dof    
    UR = fmts.PointExpression(UR_vec ,g)
    
    # Ulist.append(mpms.solveMultiscale(param, mesh, eps, op = 'BCdirich', others = [[2], uR ]))
        
    # n = FacetNormal(mesh)
    # print(feut.Integral(Ulist[i][0],mesh.dx,shape = (2,)))
    # print(feut.Integral(outer(UN[i][0],n),mesh.ds ,shape = (2,2)))
    # print(Wbasis[:,0:4].T @ Mh @ g(mesh,UN[i][0]).flatten())
    
    # print(np.sqrt(assemble(dot(Ulist[i][0],Ulist[i][0])*mesh.dx)))
    
    sigma_i = fmts.homogenisation(UN[i][0], mesh, sigma, [0,1], sigmaEps).flatten()
    
    sigma_11.append(sigma_i[0])
    # sigma_error.append(np.abs(sigma_i[0] - sigma_ref[0]))

    normUN_Uex_L2.append(normL2(UN[i][0] - UrefB, mesh.ds))
    normUR_Uex_L2.append(normL2(UR - UrefB, mesh.ds))
    normUN_UR_L2.append(normL2(UN[i][0] - UR, mesh.ds))
    # normUN_Uex_L2domain.append(normL2(UN[i][0] - Uref[0], mesh.dx))   

    normUN_Uex_Rm.append( np.linalg.norm( g(mesh,UN[i][0]).flatten()  - UrefB_dof)) 
    normUR_Uex_Rm.append(np.linalg.norm( UR_vec - UrefB_dof))
    normUN_UR_Rm.append(np.linalg.norm( g(mesh,UN[i][0]).flatten()  - UR_vec ))
    
# sigma_error_rel = np.array(sigma_error)/sigma_ref[0]
normUN_Uex_L2_rel = np.array(normUN_Uex_L2)/normUex_L2
normUR_Uex_L2_rel = np.array(normUR_Uex_L2)/normUex_L2
normUN_UR_L2_rel = np.array(normUN_UR_L2)/normUex_L2
# normUN_Uex_L2domain_rel = np.array(normUN_Uex_L2domain)/normUex_L2domain
normUN_Uex_Rm_rel = np.array(normUN_Uex_Rm)/normUex_Rm
normUR_Uex_Rm_rel = np.array(normUR_Uex_Rm)/normUex_Rm
normUN_UR_Rm_rel = np.array(normUN_UR_Rm)/normUex_Rm

cumsum = np.cumsum(sig*sig)
expectedError = np.sqrt(1-(cumsum[:-1]/cumsum[-1]))

plt.figure(1,(15,12))
plt.subplot('221')
plt.title('Absolute error fluc L2')
plt.plot(Nbasis, normUN_Uex_L2,'-o', label = 'normUN_Uex_L2')
plt.plot(Nbasis, normUR_Uex_L2,'-o', label = 'normUR_Uex_L2')
plt.plot(Nbasis, normUN_UR_L2, '-o', label = 'normUN_UR_L2')
# plt.plot(Nbasis, normUN_Uex_L2domain, '-o', label = 'normUN_Uex_L2domain')
plt.xlabel('N')
plt.ylabel('error fluc')
plt.yscale('log')
plt.grid()
plt.legend()

plt.subplot('222')
plt.title('Relative error fluc L2')
plt.plot(Nbasis, normUN_Uex_L2_rel,'-o', label = 'normUN_Uex_L2')
plt.plot(Nbasis, normUR_Uex_L2_rel,'-o', label = 'normUR_Uex_L2')
plt.plot(Nbasis, normUN_UR_L2_rel, '-o', label = 'normUN_UR_L2')
# plt.plot(Nbasis, normUN_Uex_L2domain_rel, '-o', label = 'normUN_Uex_L2domain')
plt.plot(np.arange(1,160), expectedError, '--', label = 'expected')
plt.xlabel('N')
plt.ylabel('rel error fluc')
plt.yscale('log')
plt.grid()
plt.legend()

plt.subplot('223')
plt.title('Absolute error fluc Rm')
plt.plot(Nbasis, normUN_Uex_Rm,'-o', label = 'normUN_Uex_Rm')
plt.plot(Nbasis, normUR_Uex_Rm,'-o', label = 'normUR_Uex_Rm')
plt.plot(Nbasis, normUN_UR_Rm, '-o', label = 'normUN_UR_Rm')
plt.xlabel('N')
plt.ylabel('error fluc')
plt.yscale('log')
plt.grid()
plt.legend()

plt.subplot('224')
plt.title('Relative error fluc Rm')
plt.plot(Nbasis, normUN_Uex_Rm_rel,'-o', label = 'normUN_Uex_Rm')
plt.plot(Nbasis, normUR_Uex_Rm_rel,'-o', label = 'normUR_Uex_Rm')
plt.plot(Nbasis, normUN_UR_Rm_rel, '-o', label = 'normUN_UR_Rm')
plt.plot(np.arange(1,160), expectedError, '--', label = 'expected')
plt.xlabel('N')
plt.ylabel('rel error fluc')
plt.yscale('log')
plt.grid()
plt.legend()

plt.savefig('errorDisp_case7_0.png')

plt.figure(2,(8,6))
# plt.subplot('111')
plt.title('stress')
plt.plot(Nbasis, sigma_11,'-o', label = 'stress(N)')
plt.xlabel('N')
plt.ylabel('sigma_11')
# plt.plot([Nbasis[0],Nbasis[-1]], 2*[sigma_ref[0]],'-o', label = 'stress reference')
plt.legend()
plt.grid()

# plt.subplot('222')
# plt.title('absolute error stress')
# plt.plot(Nbasis, sigma_error,'-o')
# plt.xlabel('N')
# plt.ylabel('|sigma_11(N) - sigma_11_ref|' )
# plt.yscale('log')
# plt.grid()
# plt.subplot('224')
# plt.title('relative error stress')
# plt.plot(Nbasis, sigma_error_rel,'-o')
# plt.xlabel('N')
# plt.ylabel('|sigma_11(N) - sigma_11_ref|/|sigma_11_ref|' )
# plt.yscale('log')
# plt.grid()
# plt.legend()

plt.savefig('stressError_case7_0.png')


plt.show()
