import sys, os
import numpy as np
sys.path.insert(0, '../../utils/')

import fenicsWrapperElasticity as fela
import generation_deepBoundary_lib as gdb
from dolfin import *
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import myHDF5 as myhd
import pickle

# simul_id = int(sys.argv[1])
# EpsDirection = int(sys.argv[2])

simul_id = 3
EpsDirection = 0

print("starting with", simul_id, EpsDirection)

folder_ = "/Users/felipefr/EPFL/newDLPDES/DATA/deepBoundary/data{0}/"
folder = folder_.format(simul_id)
# radFile = folder + "RVE_POD_reduced_{0}.{1}"
# nameSol = folder + 'RVE_POD_solution_red_{0}_' + str(EpsDirection) + '.{1}'
# nameC = 'C_reference_L2domain_{0}_{1}.txt'
nameMeshRef = folder + "RVE_POD_reduced_{0}.{1}".format(0,'xml')
# nameWbasis = 'Wbasis_reference_L2domain_{0}_{1}.txt'
# # nameYlist_old = 'Y_block_reference_{0}_{1}.txt'
# nameYlist = 'Y_reference_L2domain_{0}_{1}.txt'
nameInterpolation0 = 'interpolatedSolutions_{0}_{1}.txt'
nameInterpolation1 = './definitiveBasis/interpolatedSolutions_lite2_{0}_{1}.hd5'

# openFiles = []

ns = 100
# Nmax = 1000
Nblocks = 1

meshRef0 = refine(fela.EnrichedMesh(nameMeshRef))
Vref0 = VectorFunctionSpace(meshRef0,"CG", 1)

nx = 100
meshRef1 = RectangleMesh(Point(1.0/3., 1./3.), Point(2./3., 2./3.), nx, nx, diagonal='crossed')
Vref1 = VectorFunctionSpace(meshRef1,"CG", 2)

Isol0 = np.loadtxt(nameInterpolation0.format(simul_id,EpsDirection))[:ns,:]

Isol1 = myhd.loadhd5(nameInterpolation1.format(simul_id,EpsDirection),'Isol')[:ns,:]
# openFiles.append(ftemp)
# Isol1 = Isol1[:ns,:]

# error analysis all basis

dotProductL2bnd = lambda u,v, m : assemble(inner(u,v)*ds)
dotProductH10 = lambda u,v, dx : assemble(inner(grad(u),grad(v))*dx)
dotProductL2 = lambda u,v, dx : assemble(inner(u,v)*dx) 

# NbasisList = list(np.arange(1,157,5))
# NbasisList = list(np.arange(1,1000,20)) + [1000]
# NbasisList = list(np.arange(1,157,5))
NbasisList = list(np.arange(1,30,3)) + [30]

Nmax = NbasisList[-1]

nameWbasis0 = 'Wbasis_{0}_{1}_{2}.txt'
nameYlist0 = 'Y_{0}_{1}_{2}.txt'
nameTau0 = 'tau_{0}_{1}_{2}.txt'

nameWbasis1 = './definitiveBasis/Wbasis_{0}_lite2_correction_{1}_{2}.hd5'
nameYlist1 = './definitiveBasis/Y_{0}_lite2_correction_{1}_{2}.hd5'
nameTau1 = './definitiveBasis/tau_{0}_lite2_correction_{1}_{2}.hd5'


nameWbasis2 = './definitiveBasis/Wbasis_{0}_original_{1}_{2}.hd5'
nameYlist2 = './definitiveBasis/Y_{0}_original_{1}_{2}.hd5'
nameTau2 = './definitiveBasis/tau_{0}_original_solvePDE_{1}_{2}.hd5'

nameStress = folder + 'sigmaList{0}.txt'
sigmaList = np.loadtxt(nameStress.format(EpsDirection))[:ns,[0,3,1]] # flatten to voigt notation

Wbasis = {}
Ylist = {}
tau = {}
tau0 = {}

labels = {}
labels['H10_solvePDE'] = (nameWbasis2, nameYlist2, nameTau2, 'H10','', True, meshRef0, Vref0, Isol0) # bool is computeStress
labels['L2_solvePDE'] = (nameWbasis2, nameYlist2, nameTau2, 'L2','', True, meshRef0, Vref0, Isol0) # bool is computeStress
labels['L2bnd_solvePDE'] = (nameWbasis2, nameYlist2, nameTau2, 'L2bnd','',True, meshRef0, Vref0, Isol0) # bool is computeStress
labels['H10_Vrefbased'] = (nameWbasis1, nameYlist1, nameTau1, 'H10','', True, meshRef1, Vref1, Isol1) # bool is computeStress
labels['H10'] = (nameWbasis0, nameYlist0, nameTau0, 'H10','_projectLocal', True, meshRef0, Vref0, Isol0) # bool is computeStress
labels['L2'] = (nameWbasis0, nameYlist0, nameTau0, 'L2','_testPrec', True, meshRef0, Vref0, Isol0) # bool is computeStress
labels['L2bnd'] = (nameWbasis0, nameYlist0, nameTau0, 'L2bnd','_testPrec',False, meshRef0, Vref0, Isol0) # bool is computeStress



norms = {}
norms['stress'] = []
# norms['norm_L2bnd'] = ( 'ds', dotProductL2bnd)
# norms['norm_H10'] = ( 'dx', dotProductH10)
# norms['norm_L2'] = ('dx', dotProductL2)

mseErrors = {}
# pickle.dump(mseErrors,open("./figures2show/mseErrors_to1000.dat",'wb'))

# mseErrors2 = pickle.load(open("./figures2show/mseErrors_to1000.dat",'rb'))

# loading
for l, li in list(labels.items())[:4]: 
    print('loading ', l)
    Wbasis[l]= myhd.loadhd5(li[0].format(li[3],simul_id,EpsDirection),'Wbasis')
    # openFiles.append(ftemp)
    Ylist[l] = myhd.loadhd5(li[1].format(li[3], simul_id,EpsDirection), 'Ylist')
    # openFiles.append(ftemp)
    taus = myhd.loadhd5(li[2].format(li[3] + li[4],simul_id,EpsDirection),['tau','tau_0'])
    # openFiles.append(ftemp)
    
    Wbasis[l] = Wbasis[l][:Nmax,:]
    Ylist[l] = Ylist[l][:ns,:Nmax]
    tau[l] = taus[0][:ns,:Nmax,:]
    tau0[l] = taus[1][:ns,:]


for l, li in list(labels.items())[4:]: 
    print('loading ', l)
    Wbasis[l] = np.loadtxt(li[0].format(li[3],simul_id,EpsDirection))[:Nmax,:]
    Ylist[l] = np.loadtxt(li[1].format(li[3], simul_id,EpsDirection))[:ns,:Nmax]
    if(li[5]):
        tau[l] = np.loadtxt(li[2].format(li[3] + li[4],simul_id,EpsDirection))[:ns,:3*Nmax]
        tau[l] = np.reshape(tau[l],(ns,Nmax,3))
        tau0[l] = np.loadtxt(li[2].format(li[3] + li[4],simul_id, 'fixedEps' + str(EpsDirection)))[:ns,:]


# computing
for l, li in list(labels.items()):
    mseErrors[l] = {}
    for n, ni in norms.items():
        if(n=='stress'):
            computeStress = li[5]
            if(computeStress):
                mseErrors[l]['stress'] = gdb.getMSEstresses(NbasisList,Ylist[l], tau[l], tau0[l], sigmaList)

        else: 
            dy = ni[0]
            dotProduct = ni[1]
            mesh = li[6]
            dy = Measure(dy, mesh)  
            V = li[7]
            Isol = li[8]
            mseErrors[l][n] = gdb.getMSE(NbasisList,Ylist[l],Wbasis[l], Isol, V, dy, dotProduct)


kfig = 0
for n, ni in norms.items():
    kfig+=1
    plt.figure(kfig)
    plt.title("MSE " + n + "_  for $\epsilon^{0}$".format(EpsDirection + 1))
    for l, li in list(labels.items()):
        if(n!='stress' or li[5]):
            plt.plot(NbasisList[:-1],mseErrors[l][n][:-1], '-', label = l)
        
    plt.xlabel('N')
    plt.ylabel('error sqrt(mse)')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    # plt.savefig("figures2show/allAproximations_corrected_to999_{0}_eps{1}.png".format(n,EpsDirection))
