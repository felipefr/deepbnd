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

def getMRbasis(N):
    B = np.zeros((N*8,4))
    
    for i in range(N+1):
        j = 2*i
        B[j,0] = -1.0
        B[j + 1,1] = -1.0
        
    for i in range(N,2*N + 1):
        j = 2*i
        B[j,2] = -1.0
        B[j+1,3] = -1.0
        
    for i in range(2*N, 3*N +1 ):
        j = 2*i
        B[j,0] = 1.0
        B[j+1,1] = 1.0
        
    for i in range(3*N,4*N + 1):
        j = 2*(i%(4*N))
        B[j,2] = 1.0
        B[j+1,3] = 1.0    
        
    return B

S = np.loadtxt('snapshots.txt') 

# BMR = 99999.0*getMRbasis(20)
# S = np.concatenate((S,BMR,BMR,BMR,BMR,BMR,BMR),axis = 1)


# Wbasis, sig, VT = np.linalg.svd(S)

folder = "../../scratch/data/"
radFile = folder + "RVE_POD_ref_{0}.{1}"

np.random.seed(10)

offset = 0
Lx = Ly = 1.0
ifPeriodic = False 
NxL = NyL = 2
x0L = y0L = offset*Lx/(1+2*offset)
LxL = LyL = Lx/(1+2*offset)
r0 = 0.2*LxL/NxL
r1 = 0.4*LxL/NxL

ellipseData = geni.circularRegular2Regions(r0, r1, NxL, NyL, Lx, Ly, offset, ordered = True)

times = 1

lcar = 0.1*Lx/(NxL*times)

meshGMSH = gmsh.ellipseMeshRepetition(times, ellipseData, Lx, Ly , lcar, ifPeriodic)
meshGMSH.setTransfiniteBoundary(int(Lx/lcar) + 1)
print("nodes per side",  int(Lx/lcar) + 1)

meshGeoFile = radFile.format(times,'geo')
meshXmlFile = radFile.format(times,'xml')
meshMshFile = radFile.format(times,'msh')

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
# Nbasis = np.arange(159,161)


# Solving with Multiphenics
Uref = mpms.solveMultiscale(param, mesh, eps, op = 'periodic')
sigma, sigmaEps = fmts.getSigma_SigmaEps(param,mesh,eps)
sigma_ref = fmts.homogenisation(Uref[0],mesh, sigma, [0,1], sigmaEps).flatten()
g = gmts.displacementGeneratorBoundary(0.0,0.0,1.0,1.0, 21)

Nmax = np.max(Nbasis)
alpha = np.zeros(Nmax)

Wbasis,sig , Mhsqrt, Mh= fmts.pod_customised(S, Nmax, 'L2', [g,mesh]) 
# Wbasis,sig , Mhsqrt, Mh= fmts.pod_customised(S, Nmax) 

bbasis = []

start = timer()
BMR = getMRbasis(20)

for i in range(4):
    BMR[:,i] = BMR[:,i]/np.linalg.norm(BMR[:,i]) 

Wbasis = np.concatenate((BMR,Wbasis),axis = 1)

for j in range(Nmax):
    bbasis.append(fmts.PointExpression(Wbasis[:,j], g))
    alpha[j] = 0.25*assemble( inner(bbasis[-1],Uref[0])*mesh.ds )

tol = 1e-10
for j in range(4):
    alpha[j] = tol 
    
end = timer()
print('computing alpha', end - start) # Time in seconds, e.g. 5.38091952400282

UN = []

print('integral boundary : ' , Wbasis[:,0:4].T @ Mh @ g(mesh,Uref[0]).flatten())

for i,N in enumerate(Nbasis): 
    UN.append(mpms.solveMultiscale(param, mesh, eps, op = 'POD_noMR', others = [bbasis[0:N], alpha[0:N]]))
    
normUN_Uex_L2 = [] ; normUR_Uex_L2 = [] ; normUN_UR_L2 = []  ; normUN_Uex_L2domain = []
normUN_Uex_Rm = [] ; normUR_Uex_Rm = [] ; normUN_UR_Rm = [] 

sigma_11 = []
sigma_error = []  
sigma_error_rel = [] 

normL2 = lambda x,dx : np.sqrt(assemble(inner(x,x)*dx))

normUex_L2domain = normL2(Uref[0],mesh.dx) 
normUex_L2 = normL2(Uref[0],mesh.ds) 
normUex_Rm = np.linalg.norm(g(mesh,Uref[0]).flatten()) 

for i,N in enumerate(Nbasis):
    
    UR_vec = Mhsqrt @ Wbasis[:,:N] @ Wbasis[:,:N].T @  Mhsqrt @ g(mesh,Uref[0]).flatten()    
    UR = fmts.PointExpression(UR_vec ,g)
    
    # Ulist.append(mpms.solveMultiscale(param, mesh, eps, op = 'BCdirich', others = [[2], uR ]))
        
    n = FacetNormal(mesh)
    # print(feut.Integral(Ulist[i][0],mesh.dx,shape = (2,)))
    print(feut.Integral(outer(UN[i][0],n),mesh.ds ,shape = (2,2)))
    print(Wbasis[:,0:4].T @ Mh @ g(mesh,UN[i][0]).flatten())
    
    # print(np.sqrt(assemble(dot(Ulist[i][0],Ulist[i][0])*mesh.dx)))
    
    sigma_i = fmts.homogenisation(UN[i][0], mesh, sigma, [0,1], sigmaEps).flatten()
    
    sigma_11.append(sigma_i[0])
    sigma_error.append(np.abs(sigma_i[0] - sigma_ref[0]))

    normUN_Uex_L2.append(normL2(UN[i][0] - Uref[0], mesh.ds))
    normUR_Uex_L2.append(normL2(UR - Uref[0], mesh.ds))
    normUN_UR_L2.append(normL2(UN[i][0] - UR, mesh.ds))
    normUN_Uex_L2domain.append(normL2(UN[i][0] - Uref[0], mesh.dx))   

    normUN_Uex_Rm.append( np.linalg.norm( g(mesh,UN[i][0]).flatten()  - g(mesh,Uref[0]).flatten()) )
    normUR_Uex_Rm.append(np.linalg.norm( UR_vec   - g(mesh,Uref[0]).flatten()))
    normUN_UR_Rm.append(np.linalg.norm( g(mesh,UN[i][0]).flatten()  - UR_vec ))
    
sigma_error_rel = np.array(sigma_error)/sigma_ref[0]
normUN_Uex_L2_rel = np.array(normUN_Uex_L2)/normUex_L2
normUR_Uex_L2_rel = np.array(normUR_Uex_L2)/normUex_L2
normUN_UR_L2_rel = np.array(normUN_UR_L2)/normUex_L2
normUN_Uex_L2domain_rel = np.array(normUN_Uex_L2domain)/normUex_L2domain
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
plt.plot(Nbasis, normUN_Uex_L2domain, '-o', label = 'normUN_Uex_L2domain')
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
plt.plot(Nbasis, normUN_Uex_L2domain_rel, '-o', label = 'normUN_Uex_L2domain')
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

plt.savefig('errorDisp.png')

plt.figure(2,(15,10))
plt.subplot('121')
plt.title('stress')
plt.plot(Nbasis, sigma_11,'-o', label = 'stress(N)')
plt.xlabel('N')
plt.ylabel('sigma_11')
plt.plot([Nbasis[0],Nbasis[-1]], 2*[sigma_ref[0]],'-o', label = 'stress reference')
plt.legend()
plt.grid()

plt.subplot('222')
plt.title('absolute error stress')
plt.plot(Nbasis, sigma_error,'-o')
plt.xlabel('N')
plt.ylabel('|sigma_11(N) - sigma_11_ref|' )
plt.yscale('log')
plt.grid()
plt.subplot('224')
plt.title('relative error stress')
plt.plot(Nbasis, sigma_error_rel,'-o')
plt.xlabel('N')
plt.ylabel('|sigma_11(N) - sigma_11_ref|/|sigma_11_ref|' )
plt.yscale('log')
plt.grid()
plt.legend()

plt.savefig('stressError.png')


plt.show()
