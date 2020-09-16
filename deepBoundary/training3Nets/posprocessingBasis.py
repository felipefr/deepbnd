import sys, os
import numpy as np
sys.path.insert(0, '../../utils/')

import fenicsWrapperElasticity as fela
import generation_deepBoundary_lib as gdb
from dolfin import *
from timeit import default_timer as timer
import matplotlib.pyplot as plt

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

# dotProduct = lambda u,v, m : assemble(inner(u,v)*ds)
# dotProduct = lambda u,v, m : assemble(inner(grad(u),grad(v))*m.dx)
# dotProduct = lambda u,v, dx : assemble(inner(grad(u),grad(v))*dx)
dotProduct = lambda u,v, dx : assemble(inner(u,v)*dx) 
# dotProduct = lambda u,v, dx : np.dot(u.vector()[:],v.vector()[:]) 

ns = 1000
Nmax = 1000
Nblocks = 1

meshRef = refine(fela.EnrichedMesh(nameMeshRef))
Vref = VectorFunctionSpace(meshRef,"CG", 1)
dxRef = Measure('dx', meshRef) 
dsRef = Measure('ds', meshRef) 

Isol = np.loadtxt(nameInterpolation.format(simul_id,EpsDirection))

# Computing basis 
# C = np.loadtxt(nameC.format(simul_id,EpsDirection))
# Wbasis = np.loadtxt(nameWbasis.format(simul_id,EpsDirection))[:Nmax,:]
# Ylist = np.loadtxt(nameYlist.format(simul_id,EpsDirection))

# error analysis
# NbasisList = list(np.arange(1,1001,10)) + [1000]

# mseErrors = gdb.getMSE(NbasisList,Ylist[:100,:],Wbasis, Isol[:100,:], Vref, dxRef, dotProduct)

# sig, U = np.linalg.eigh(C)

# asort = np.argsort(sig)
# sig = sig[asort[::-1]]
# U = U[:,asort[::-1]]

# acsum = np.sqrt(np.cumsum(sig[::-1])[::-1])

# plt.figure(3)
# plt.title("MSE for 100 first Snapshots (based on L2 domain)")
# plt.plot(NbasisList[:],mseErrors[:], '-o')
# plt.plot(np.arange(1,1001),acsum,'k','--')
# plt.xlabel('N')
# plt.ylabel('error sqrt(mse)')
# plt.yscale('log')
# plt.legend(['empirical','theory'])
# plt.grid()
# plt.savefig("errorVerification_100_L2domain_untilEnd.png")
# plt.plot()

# Basis functions visualisations
# u = Function(Vref) 

# u.vector().set_local(Wbasis[0,:])
# plt.figure(1,(7,11))
# plt.suptitle("Reduced basis for $\epsilon^3$ (based on $L^2(\Omega_{\mu})$)",fontsize=20)
# plt.subplot('321')
# plt.title(r"$\xi^1_1$", fontsize = 15)
# plot(u[0])
# plt.subplot('322')
# plt.title(r"$\xi^1_2$", fontsize = 15)
# plot(u[1])

# u.vector().set_local(Wbasis[1,:])
# plt.subplot('323')
# plt.title(r"$\xi^2_1$",fontsize = 15)
# plot(u[0])
# plt.subplot('324')
# plt.title(r"$\xi^2_2$", fontsize = 15)
# plot(u[1])

# u.vector().set_local(Wbasis[2,:])
# plt.subplot('325')
# plt.title(r"$\xi^3_1$",fontsize = 15)
# plot(u[0])
# plt.subplot('326')
# plt.title(r"$\xi^3_2$",fontsize = 15)
# plot(u[1])

# plt.tight_layout(rect=[0, 0.03, 1, 0.97])
# plt.savefig("basis_L2domain_eps3.pdf")
# plt.savefig("basis_L2domain_eps3.png")

# plt.plot()



# error analysis all basis

dotProductL2bnd = lambda u,v, m : assemble(inner(u,v)*ds)
dotProductH10 = lambda u,v, dx : assemble(inner(grad(u),grad(v))*dx)
dotProductL2 = lambda u,v, dx : assemble(inner(u,v)*dx) 

# NbasisList = list(np.arange(1,157,5))
NbasisList = list(np.arange(1,1000,20)) + [1000]

nameWbasis = 'Wbasis_reference_{0}{1}_{2}.txt'
nameYlist = 'Y_reference_{0}{1}_{2}.txt'

WbasisL2bnd = np.loadtxt(nameWbasis.format('L2_',simul_id,EpsDirection))[:Nmax,:]
YlistL2bnd = np.loadtxt(nameYlist.format('L2_', simul_id,EpsDirection))

WbasisH10 = np.loadtxt(nameWbasis.format('', simul_id,EpsDirection))[:Nmax,:]
YlistH10 = np.loadtxt(nameYlist.format('', simul_id,EpsDirection))

WbasisL2 = np.loadtxt(nameWbasis.format('L2domain_', simul_id,EpsDirection))[:Nmax,:]
YlistL2 = np.loadtxt(nameYlist.format('L2domain_', simul_id,EpsDirection))

nameTau = 'tau_{0}_{1}_{2}.txt'
tauL2bnd = np.loadtxt(nameTau.format('L2_testPrec',simul_id,EpsDirection))
tau0L2bnd = np.loadtxt(nameTau.format('L2_testPrec',simul_id, 'fixedEps' + str(EpsDirection)))

tauL2 = np.loadtxt(nameTau.format('L2domain_testPrec',simul_id, EpsDirection))
tau0L2 = np.loadtxt(nameTau.format('L2domain_testPrec',simul_id, 'fixedEps' + str(EpsDirection)))

tauH10 = np.loadtxt(nameTau.format('H10_projectlocal',simul_id, EpsDirection))
tau0H10 = np.loadtxt(nameTau.format('H10_projectlocal',simul_id, 'fixedEps' + str(EpsDirection)))

nameStress = folder + 'sigmaList{0}.txt'
sigmaList = np.loadtxt(nameStress.format(EpsDirection))[:100,[0,3,1]] # flatten to voigt notation

mseErrors = {}
mseErrors['L2_norm_L2'] = gdb.getMSE(NbasisList,YlistL2bnd[:100,:],WbasisL2bnd, Isol[:100,:], Vref, dsRef, dotProductL2bnd)
mseErrors['L2domain_norm_L2'] = gdb.getMSE(NbasisList,YlistL2[:100,:],WbasisL2, Isol[:100,:], Vref, dsRef, dotProductL2bnd)
mseErrors['H10_norm_L2'] = gdb.getMSE(NbasisList,YlistH10[:100,:],WbasisH10, Isol[:100,:], Vref, dsRef, dotProductL2bnd)

mseErrors['L2_norm_H10'] = gdb.getMSE(NbasisList,YlistL2bnd[:100,:],WbasisL2bnd, Isol[:100,:], Vref, dxRef, dotProductH10)
mseErrors['L2domain_norm_H10'] = gdb.getMSE(NbasisList,YlistL2[:100,:],WbasisL2, Isol[:100,:], Vref, dxRef, dotProductH10)
mseErrors['H10_norm_H10'] = gdb.getMSE(NbasisList,YlistH10[:100,:],WbasisH10, Isol[:100,:], Vref, dxRef, dotProductH10)

mseErrors['L2_norm_L2domain'] = gdb.getMSE(NbasisList,YlistL2bnd[:100,:],WbasisL2bnd, Isol[:100,:], Vref, dxRef, dotProductL2)
mseErrors['L2domain_norm_L2domain'] = gdb.getMSE(NbasisList,YlistL2[:100,:],WbasisL2, Isol[:100,:], Vref, dxRef, dotProductL2)
mseErrors['H10_norm_L2domain'] = gdb.getMSE(NbasisList,YlistH10[:100,:],WbasisH10, Isol[:100,:], Vref, dxRef, dotProductL2)

# WbasisH1 = np.loadtxt(nameWbasis.format('H1_', simul_id,EpsDirection))[:Nmax,:]
# YlistH1 = np.loadtxt(nameYlist.format('H1_', simul_id,EpsDirection))
# mseErrors['H1_norm_L2'] = gdb.getMSE(NbasisList,YlistH1[:100,:],WbasisH1, Isol[:100,:], Vref, dsRef, dotProductL2bnd)
# mseErrors['H1_norm_H10'] = gdb.getMSE(NbasisList,YlistH1[:100,:],WbasisH1, Isol[:100,:], Vref, dxRef, dotProductH10)
# mseErrors['H1_norm_L2domain'] = gdb.getMSE(NbasisList,YlistH1[:100,:],WbasisH1, Isol[:100,:], Vref, dxRef, dotProductL2)

# mseErrors['L2_norm_stress'] = gdb.getMSEstresses(NbasisList,YlistL2bnd[:100,:], tauL2bnd, tau0L2bnd, sigmaList)
mseErrors['L2domain_norm_stress'] = gdb.getMSEstresses(NbasisList,YlistL2[:100,:], tauL2[:100,:], tau0L2[:100,:], sigmaList[:100,:])
mseErrors['H10_norm_stress'] = gdb.getMSEstresses(NbasisList,YlistH10[:100,:], tauH10[:100,:], tau0H10[:100,:] , sigmaList[:100,:])

plt.figure(3)
plt.title("MSE ($H^1_0(\Omega)$) for $\epsilon^{0}$".format(EpsDirection + 1))
plt.plot(NbasisList[:],mseErrors['L2_norm_H10'], '-')
plt.plot(NbasisList[:],mseErrors['L2domain_norm_H10'], '-')
plt.plot(NbasisList[:],mseErrors['H10_norm_H10'], '-')
# plt.plot(NbasisList[:],mseErrors['H1_norm_H10'], '-')
plt.xlabel('N')
plt.ylabel('error sqrt(mse)')
plt.yscale('log')
plt.legend(['L2bnd','L2','H10','H1'])
plt.grid()
# plt.savefig("allAproximations_H10_eps{0}.png".format(EpsDirection + 1))


plt.figure(4)
plt.title("MSE ($L^2(\Omega)$) for $\epsilon^{0}$".format(EpsDirection + 1))
plt.plot(NbasisList[:],mseErrors['L2_norm_L2domain'], '-')
plt.plot(NbasisList[:],mseErrors['L2domain_norm_L2domain'], '-')
plt.plot(NbasisList[:],mseErrors['H10_norm_L2domain'], '-')
# plt.plot(NbasisList[:],mseErrors['H1_norm_L2domain'], '-')
plt.xlabel('N')
plt.ylabel('error sqrt(mse)')
plt.yscale('log')
plt.legend(['L2bnd','L2','H10','H1'])
plt.grid()
# plt.savefig("allAproximations_L2domain_eps{0}.png".format(EpsDirection + 1))

plt.figure(5)
plt.title("MSE ($L^2(\partial \Omega)$) for $\epsilon^{0}$".format(EpsDirection + 1))
plt.plot(NbasisList[:-1],mseErrors['L2_norm_L2'][:-1], '-')
plt.plot(NbasisList[:-1],mseErrors['L2domain_norm_L2'][:-1], '-')
plt.plot(NbasisList[:-1],mseErrors['H10_norm_L2'][:-1], '-')
# plt.plot(NbasisList[:-1],mseErrors['H1_norm_L2'][:-1], '-')
plt.xlabel('N')
plt.ylabel('error sqrt(mse)')
plt.yscale('log')
plt.legend(['L2bnd','L2','H10','H1'])
plt.grid()
# plt.savefig("allAproximations_L2_eps{0}.png".format(EpsDirection + 1))

plt.figure(6)
plt.title("MSE (stress) for $\epsilon^{0}$".format(EpsDirection + 1))
# plt.plot(NbasisList[:-1],mseErrors['L2_norm_stress'][:-1], '-')
plt.plot(NbasisList[:],mseErrors['L2domain_norm_stress'][:], '-')
plt.plot(NbasisList[:],mseErrors['H10_norm_stress'][:], '-')
plt.xlabel('N')
plt.ylabel('error sqrt(mse)')
plt.yscale('log')
# plt.legend(['L2bnd','L2','H10','H1'])
plt.legend(['L2','H10'])
plt.grid()
# plt.savefig("allAproximations_stressComplete_eps{0}.png".format(EpsDirection + 1))

plt.plot()