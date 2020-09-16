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

def getMesh(meshGMSH, label, radFile):
    meshGeoFile = radFile.format(label,'geo')
    meshXmlFile = radFile.format(label,'xml')
    meshMshFile = radFile.format(label,'msh')
    
    meshGMSH.write(meshGeoFile,'geo')
    os.system('gmsh -2 -algo del2d -format msh2 ' + meshGeoFile)
    
    os.system('dolfin-convert {0} {1}'.format(meshMshFile, meshXmlFile))
    
    mesh = fela.EnrichedMesh(meshXmlFile)
    
    return mesh


S = np.loadtxt('snapshots_periodic_balanced_new.txt')
EpsFluc = np.loadtxt('EpsPer.txt')

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

facL = Lx/LxL

S = facL*S

ellipseData = np.loadtxt(folder + 'ellipseData_' + str(idEllipse) + '.txt')[:NL]

ellipseData[:,0] = facL*(ellipseData[:,0] - x0L) 
ellipseData[:,1] = facL*(ellipseData[:,1] - y0L) 
ellipseData[:,2] = facL*ellipseData[:,2] 

meshGMSH = gmsh.ellipseMeshRepetition(times, ellipseData, Lx, Ly , lcar, ifPeriodic)
# meshGMSH = gmsh.ellipseMesh2(ellipseData[:NL], x0L, y0L, LxL, LyL , lcar, ifPeriodic)
meshGMSH.setTransfiniteBoundary(NpLxL)

mesh = getMesh(meshGMSH,'reduced', radFile)


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

Nbasis = np.array([ 3 + int(i**2.04) for i in range(1,12)])

eps = eps0 + EpsFluc[idEllipse,:].reshape((2,2))
eps = 0.5*(eps + eps.T)

g = gmts.displacementGeneratorBoundary(0.0,0.0, Lx, Ly, NpLxL)
UrefB_dof = S[:,idEllipse]
UrefB = fmts.PointExpression(UrefB_dof, g)

UrefComplete = mpms.solveMultiscale(param, mesh, eps, op = 'BCdirich_PointExpression', others = [[2], UrefB ])

sigma, sigmaEps = fmts.getSigma_SigmaEps(param,mesh,eps)

sigma_ref = fmts.homogenisation(UrefComplete[0], mesh, sigma, [0,1], sigmaEps).flatten()

Nmax = np.max(Nbasis)
alpha = np.zeros(Nmax)

Wbasis,sig , Mhsqrt, Mh= fmts.pod_customised(S, Nmax, 'L2', [g,mesh]) 
# Wbasis,sig, Mhsqrt, Mh = fmts.pod_customised(S, Nmax, 'l2') 

bbasis = []

start = timer()
meas_ds = assemble( Constant(1.0)*mesh.ds )

for j in range(Nmax):
    bbasis.append(fmts.PointExpression(Wbasis[:,j], g))
    alpha[j] = assemble( inner(bbasis[-1],UrefB)*mesh.ds )/meas_ds

end = timer()
print('time for computing alphas', end - start) # Time in seconds, e.g. 5.38091952400282

UN = []


for i,N in enumerate(Nbasis): 
    UN.append(mpms.solveMultiscale(param, mesh, eps, op = 'POD', others = [bbasis[0:N], alpha[0:N]]))
    
normUN_Uex_L2 = [] ; normUR_Uex_L2 = [] ; normUN_UR_L2 = []  ; normUN_Uex_L2domain = []
normUN_Uex_Rm = [] ; normUR_Uex_Rm = [] ; normUN_UR_Rm = []  ; normUR_Uex_L2domain = []

sigma_11 = []
sigma_error = []  
sigma_11_strong = []
sigma_error_strong = []  

normL2 = lambda x,dx : np.sqrt(assemble(inner(x,x)*dx))

# normUex_L2domain = np.sqrt( normL2(UrefComplete[0],mesh2.dx(0))**2.0 + normL2(UrefComplete[0],mesh2.dx(1))**2.0 )
normUex_L2domain = normL2(UrefComplete[0],mesh.dx) 
normUex_L2 = normL2(UrefB,mesh.ds) 
normUex_Rm = np.linalg.norm(UrefB_dof) 

UNstrong = []

for i,N in enumerate(Nbasis):
    UR_vec = Mhsqrt @ Wbasis[:,:N] @ Wbasis[:,:N].T @  Mhsqrt @ UrefB_dof    
    
    UR = fmts.PointExpression(UR_vec ,g)
    
    UNstrong.append(mpms.solveMultiscale(param, mesh, eps, op = 'BCdirich_PointExpression', others = [[2], UR ]))
        
    normal = FacetNormal(mesh)
    print(feut.Integral(UN[i][0],mesh.dx,shape = (2,)))
    print(feut.Integral(outer(UN[i][0],normal),mesh.ds ,shape = (2,2)))
    print(feut.Integral(UNstrong[-1][0],mesh.dx,shape = (2,)))
    print(feut.Integral(outer(UNstrong[-1][0],normal),mesh.ds ,shape = (2,2)))

    # print(Wbasis[:,0:4].T @ Mh @ g(mesh,UN[i][0]).flatten())
    
    # print(np.sqrt(assemble(dot(Ulist[i][0],Ulist[i][0])*mesh.dx)))
    
    sigma_i_strong = fmts.homogenisation(UNstrong[-1][0], mesh, sigma, [0,1], sigmaEps).flatten()
    
    sigma_i = fmts.homogenisation(UN[i][0], mesh, sigma, [0,1], sigmaEps).flatten()
    
    sigma_11_strong.append(sigma_i_strong[0])
    
    sigma_11.append(sigma_i[0])
    
    sigma_error.append(np.abs(sigma_i[0] - sigma_ref[0]))

    sigma_error_strong.append(np.abs(sigma_i_strong[0] - sigma_ref[0]))
    
    plt.figure(i,(10,18))
    plt.subplot('321')
    plot(UrefComplete[0][0])
    plt.subplot('322')
    plot(UrefComplete[0][1])
    plt.subplot('323')
    plot(UN[i][0][0])
    plt.subplot('324')
    plot(UN[i][0][1])
    plt.subplot('325')
    plot(UNstrong[-1][0][0])
    plt.subplot('326')
    plot(UNstrong[-1][0][1])

    normUN_Uex_L2.append(normL2(UN[i][0] - UrefB, mesh.ds))
    normUR_Uex_L2.append(normL2(UR - UrefB, mesh.ds))
    normUN_UR_L2.append(normL2(UN[i][0] - UR, mesh.ds))
    normUN_Uex_L2domain.append(normL2(UN[i][0] - UrefComplete[0], mesh.dx))   
    normUR_Uex_L2domain.append(normL2(UNstrong[-1][0] - UrefComplete[0], mesh.dx))

    normUN_Uex_Rm.append( np.linalg.norm( g(mesh,UN[i][0]).flatten()  - UrefB_dof)) 
    normUR_Uex_Rm.append(np.linalg.norm( UR_vec - UrefB_dof))
    normUN_UR_Rm.append(np.linalg.norm( g(mesh,UN[i][0]).flatten()  - UR_vec ))
    
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

plt.figure(1,(15,12))
plt.subplot('221')
plt.title('Absolute error fluc L2')
plt.plot(Nbasis, normUN_Uex_L2,'-o', label = 'normUN_Uex_L2')
plt.plot(Nbasis, normUR_Uex_L2,'-o', label = 'normUR_Uex_L2')
plt.plot(Nbasis, normUN_UR_L2, '-o', label = 'normUN_UR_L2')
plt.plot(Nbasis, normUN_Uex_L2domain, '-o', label = 'normUN_Uex_L2domain')
plt.plot(Nbasis, normUR_Uex_L2domain, '-o', label = 'normUR_Uex_L2domain')
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
plt.plot(Nbasis, normUR_Uex_L2domain_rel, '-o', label = 'normURe_Uex_L2domain')
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

plt.figure(2,(10,10))
plt.subplot('121')
plt.title('stress')
plt.plot(Nbasis, sigma_11,'-o', label = 'stress(N)')
plt.plot(Nbasis, sigma_11_strong,'-o', label = 'stress(R(N))')
plt.xlabel('N')
plt.ylabel('sigma_11')
plt.plot([Nbasis[0],Nbasis[-1]], 2*[sigma_ref[0]],'-o', label = 'stress reference')
plt.legend()
plt.grid()

plt.subplot('222')
plt.title('absolute error stress')
plt.plot(Nbasis, sigma_error,'-o',label = 'stress(N)')
plt.plot(Nbasis, sigma_error_strong,'-o', label = 'stress(R(N))')
plt.xlabel('N')
plt.ylabel('|sigma_11(N) - sigma_11_ref|' )
plt.yscale('log')
plt.grid()
plt.subplot('224')
plt.title('relative error stress')
plt.plot(Nbasis, sigma_error_rel,'-o', label = 'stress(N)')
plt.plot(Nbasis, sigma_error_strong_rel,'-o', label = 'stress(R(N))')
plt.xlabel('N')
plt.ylabel('|sigma_11(N) - sigma_11_ref|/|sigma_11_ref|' )
plt.yscale('log')
plt.grid()
plt.legend()

plt.savefig('stressError_case7_0.png')


plt.show()
