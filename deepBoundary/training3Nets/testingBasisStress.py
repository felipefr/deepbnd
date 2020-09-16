import sys, os
import numpy as np
sys.path.insert(0, '../../utils/')

import fenicsWrapperElasticity as fela
import generation_deepBoundary_lib as gdb
from dolfin import *
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import multiphenicsMultiscale as mpms
import elasticity_utils as elut
import fenicsMultiscale as fmts
import ioFenicsWrappers as iofe
from myCoeffClass import *
import fenicsUtils as feut

# simul_id = int(sys.argv[1])
# EpsDirection = int(sys.argv[2])

simul_id = 3
EpsDirection = 0

print("starting with", simul_id, EpsDirection)

folder_ = "/Users/felipefr/EPFL/newDLPDES/DATA/deepBoundary/data{0}/"
folder = folder_.format(simul_id)
radFile = folder + "RVE_POD_reduced_{0}.{1}"
nameSol = folder + 'RVE_POD_solution_red_{0}_' + str(EpsDirection) + '.{1}'
nameC = 'C_reference_L2domain_{0}_{1}.txt'
nameMeshRef = folder + "RVE_POD_reduced_{0}.{1}".format(0,'xml')
nameWbasis = 'Wbasis_reference_L2domain_{0}_{1}.txt'
# nameYlist_old = 'Y_block_reference_{0}_{1}.txt'
nameYlist = 'Y_reference_L2domain_{0}_{1}.txt'
nameInterpolation = 'interpolatedSolutions_{0}_{1}.txt'


meshRef = refine(fela.EnrichedMesh(nameMeshRef))
Vref = VectorFunctionSpace(meshRef,"CG", 1)
dxRef = Measure('dx', meshRef) 
dsRef = Measure('ds', meshRef) 

Isol = np.loadtxt(nameInterpolation.format(simul_id,EpsDirection))

# error analysis all basis

# dotProductL2bnd = lambda u,v, m : assemble(inner(u,v)*ds)
# dotProductH10 = lambda u,v, dx : assemble(inner(grad(u),grad(v))*dx)
# dotProductL2 = lambda u,v, dx : assemble(inner(u,v)*dx) 

NbasisList = list(np.arange(1,157,5))

nameWbasis = 'Wbasis_reference_{0}{1}_{2}.txt'
nameYlist = 'Y_reference_{0}{1}_{2}.txt'


WbasisH10 = np.loadtxt(nameWbasis.format('', simul_id,EpsDirection))
YlistH10 = np.loadtxt(nameYlist.format('', simul_id,EpsDirection))


nameTau = 'tau_{0}_{1}_{2}.txt'


tauH10 = np.loadtxt(nameTau.format('H10_complete',simul_id, EpsDirection))
tau0H10 = np.loadtxt(nameTau.format('H10_complete',simul_id, 'fixedEps' + str(EpsDirection)))

nameStress = folder + 'sigmaList{0}.txt'
sigmaList = np.loadtxt(nameStress.format(EpsDirection))[:100,[0,3,1]] # flatten to voigt notation

# mseErrors = {}
# mseErrors['H10_norm_L2'] = gdb.getMSE(NbasisList,YlistH10[:100,:],WbasisH10, Isol[:100,:], Vref, dsRef, dotProductL2bnd)

# mseErrors['H10_norm_H10'] = gdb.getMSE(NbasisList,YlistH10[:100,:],WbasisH10, Isol[:100,:], Vref, dxRef, dotProductH10)

# mseErrors['H10_norm_L2domain'] = gdb.getMSE(NbasisList,YlistH10[:100,:],WbasisH10, Isol[:100,:], Vref, dxRef, dotProductL2)


# mseErrors['H10_norm_stress'] = gdb.getMSEstresses(NbasisList,YlistH10[:100,:], tauH10, tau0H10 , sigmaList)

ns0 = 9
ns = ns0 + 1
N = 1000
tau = tauH10[:ns,:3*N].reshape((ns,N,3))

stressR = np.einsum('ijk,ij->ik',tau[ns0:ns,:N,:],YlistH10[ns0:ns,:N]) + tau0H10[ns0:ns,:]
print(sigmaList[ns0:ns,:] - stressR)

# the hard way 
ten2voigt = lambda A : np.array([A[0,0],A[1,1],0.5*(A[0,1] + A[1,0])])
param = [10.0, 0.3, 0.1]
nameMeshPrefix = folder + "RVE_POD_reduced_{0}.{1}"
nameMeshRef = nameMeshPrefix.format(0,'xml')
EpsFlucPrefix = folder + 'EpsList_{0}.txt'

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

EpsFluc = np.loadtxt(EpsFlucPrefix.format(EpsDirection))

basis = Function(Vref)

op = 'direct'

if(op == 'direct'):
    transformBasis = lambda w,W, epsbar : interpolate(w,W)
elif(op == 'solvePDE_BC'):
    transformBasis = lambda w,W, epsbar : mpms.solveMultiscale(param[0:2,:], W.mesh(), epsbar, op = 'BCdirich_lag', others = [w])[0]
else:
    print('Wrong option ', op)
    input()


mesh = fela.EnrichedMesh(nameMeshPrefix.format(ns0,'xml'))
V = VectorFunctionSpace(mesh,"CG", 1)

epsL = (EpsUnits + EpsFluc[ns0,:]).reshape((2,2))
sigmaL, sigmaEpsL = fmts.getSigma_SigmaEps(param,mesh,epsL)
sigmaL0, sigmaEpsL0 = fmts.getSigma_SigmaEps(param,mesh,np.zeros((2,2)))

basis.vector().set_local(np.zeros(Vref.dim()))
Ibasis = transformBasis(basis,V,epsL)
sigma0 = ten2voigt(fmts.homogenisation(Ibasis, mesh, sigmaL, [0,1], sigmaEpsL))

# Wbasis[:N,:].T@Ylist[i,:N]
basis.vector().set_local( WbasisH10[:N,:].T@YlistH10[ns0,:N])
Ibasis = transformBasis(basis,V, np.zeros((2,2)))
sigma = sigma0 +  ten2voigt(fmts.homogenisation(Ibasis, mesh, sigmaL0, [0,1]))

print(sigmaList[ns0,:] - sigma)

print(stressR - sigma)

basis.vector().set_local(Isol[ns0,:])
Ibasis = transformBasis(basis,V, np.zeros((2,2)))
sigmaReinterp = sigma0 + ten2voigt(fmts.homogenisation(Ibasis, mesh, sigmaL0, [0,1]))

print(stressR - sigmaReinterp)
print(sigma- sigmaReinterp)
print(sigmaList[ns0,:] - sigmaReinterp)


uBC = gdb.recoverFenicsSimulation(nameMeshPrefix.format(ns0,'xml'), nameSol.format(ns0,'h5'))


meshnew = fela.EnrichedMesh(nameMeshPrefix.format(ns0,'xml'))
meshnew.V['u'] = VectorFunctionSpace(meshnew, 'CG', 1)
W = meshnew.V['u']
WS  = FunctionSpace(meshnew, 'DG', 0)
# W = VectorFunctionSpace(meshnew, 'CG', 2)
uOrig = mpms.solveMultiscale(param[0:2,:], meshnew, epsL, op = 'BCdirich_lag', others = [uBC])[0]

# mesh0 = refine(fela.EnrichedMesh(nameMeshPrefix.format(0,'xml')))
nx = 100
mesh0 = RectangleMesh(Point(1.0/3., 1./3.), Point(2./3., 2./3.), nx, nx, diagonal='crossed')
W0 = VectorFunctionSpace(mesh0, 'CG', 2)
W0S = FunctionSpace(mesh0, 'CG', 4)



IuOrig = interpolate(uOrig, W0)
IIuOrig = interpolate(IuOrig, W)

# PGIuOrig = project(sigmaL(IuOrig)[0,0], WS)

e = Function(W)
e.vector().set_local(IIuOrig.vector()[:] - uOrig.vector()[:])
iofe.postProcessing_complete(e , 'error.xdmf', labels = ['u', 'vonMises', 'lame'], param = param , rename = False)
# iofe.postProcessing_complete(IIuOrig, 'IIuOrig.xdmf', labels = ['u', 'vonMises', 'lame'], param = param)
# PGuOrig = iofe.local_project(grad(uOrig), WT)

dxx0 = Measure('dx', mesh0)
dxx = Measure('dx', meshnew) 
dss = Measure('ds', meshnew) 
# e = grad(uOrig) - grad(IIuOrig)
# e = PGuOrig - PGIuOrig
# print(assemble( inner(e,e)*dxx))
# print(assemble( inner(grad(e),grad(e))*dxx))
# print(assemble( inner(e,e)*dss))

# u = uOrig
mesh = meshnew
materials = mesh.subdomains.array().astype('int32')
materials -= np.min(materials)
lame = getMyCoeff(materials , param, op = 'python') 
lame0_ = iofe.local_project(lame[0],WS)
lame1_ = iofe.local_project(lame[1],WS)
lame_ = as_vector((lame0_,lame1_))

lame0 = interpolate(iofe.local_project(lame[0],WS), W0S)
lame1 = interpolate(iofe.local_project(lame[1],WS), W0S)
lame_0 = as_vector((lame0,lame1))
sigma = lambda u: fela.sigmaLame(u,lame) 
sigma0 = lambda u: fela.sigmaLame(u,lame_0) 

# sigmaEps = lambda Eps: lame[0]*tr(Eps)*Identity(2) + lame[1]*(Eps + Eps.T)
# Vsig = TensorFunctionSpace(mesh, "CG", 1, (2,2))
# stress0 = Function(Vsig, name="Stress")
# stress0.assign(iofe.local_project(sigma(u),Vsig)) 
# Eps0 = Function(Vsig, name="Eps")
# Eps0.assign(iofe.local_project(grad(IIuOrig),Vsig)) 

# Veps = TensorFunctionSpace(mesh0, "CG", 1, (2,2))
# IEps = project(grad(IuOrig),Veps)
# # IEps = Function(Veps, name="Eps")
# # IEps.assign(iofe.local_project(,Veps)) 
# IIEps = interpolate(IEps, Vsig)

# # norm_stress.assign(iofe.local_project(norm_stress_, Vsig))

# # print(assemble(norm_stress*dxx))
# print(assemble(inner(stress0,stress0)*dxx))
# print(assemble(inner(sigma(u),sigma(u))*dxx))
# print(assemble(inner(sigma(IIuOrig),sigma(IIuOrig))*dxx))
# print(assemble(inner(sigmaEps(IIEps),sigmaEps(IIEps))*dxx))
# print(assemble(inner(sigmaEps(Eps0),sigmaEps(Eps0))*dxx))

# print(assemble(inner(grad(uOrig),grad(uOrig))*dxx))
# print(assemble(inner(sigma(u),sigma(u))*dxx0) - assemble(inner(grad(uOrig),grad(uOrig))*dxx))


# plt.figure(1)
# c = plot((sigma(IIuOrig) - sigma(u))[0,0], mode='color')
# plt.colorbar(c)
# plt.show()

dxx = Measure("dx", meshnew)
print(assemble(inner(sigma(uOrig),sigma(uOrig))*dxx))
print(assemble(inner(sigma(IIuOrig),sigma(IIuOrig))*dxx))

dx0 = Measure("dx", mesh0)
print(assemble(inner(sigma0(IuOrig),sigma0(IuOrig))*dx0))

# dxx = Measure("dx", meshnew)
print(assemble(sigma(uOrig)[0,0]*dxx))
print(assemble(sigma(IIuOrig)[0,0]*dxx))

# dxx = Measure("dx", mesh0)
print(assemble(sigma0(IuOrig)[0,0]*dx0))

print(np.linalg.norm(feut.Integral(sigma0(IuOrig),dx0,(2,2)) - feut.Integral(sigma(uOrig),dxx, (2,2)), 'fro'))
print(np.linalg.norm(feut.Integral(sigma(IIuOrig),dxx,(2,2)) - feut.Integral(sigma(uOrig),dxx, (2,2)) , 'fro'))
# print(norm_(sigma(IIuOrig),sigma(uOrig), dx0, dxx ))

plt.figure(2)
c2 = plot(sigma0(IuOrig)[0,0], mode='color')
plt.colorbar(c2)
plt.show()