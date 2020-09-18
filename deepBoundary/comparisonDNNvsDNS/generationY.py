import sys, os
import numpy as np
sys.path.insert(0, '../../utils/')
sys.path.insert(0, '../training3Nets/')

import fenicsWrapperElasticity as fela
import generation_deepBoundary_lib as gdb
from dolfin import *
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import myHDF5 as myhd
# simul_id = int(sys.argv[1])
# EpsDirection = int(sys.argv[2])

simul_id = 3
EpsDirection = 0

print("starting with", simul_id, EpsDirection)
DATAfolder = "/Users/felipefr/EPFL/newDLPDES/DATA/"
folderBasis = DATAfolder + "deepBoundary/training3Nets/definitiveBasis/"
folderTest = "./"
folderTestDATA = DATAfolder + "deepBoundary/comparisonDNNvsDNS/coarseP1/"

formulationLabel = "L2bnd"
formulationLabel2 = "L2bnd_original_SolvePDE"
formulationLabel3 = "L2bnd_original"
folder_ = "/Users/felipefr/EPFL/newDLPDES/DATA/deepBoundary/data{0}/"
folder = folder_.format(simul_id)
radFile = folderTestDATA + "RVE_POD_reduced_ref_{0}.{1}"
nameSol = folderTestDATA + 'RVE_POD_solRed_periodic_offset0_{0}.{1}'
nameC = folderBasis + 'C_{0}_{1}_{2}.hd5'
nameMeshPrefix = folderTestDATA + "RVE_POD_reduced_ref_{0}.{1}"
nameWbasis = folderBasis + 'Wbasis_{0}_{1}_{2}.hd5'
nameYlist = folderTest + 'Y_{0}_{1}_{2}.hd5'
nameInterpolation = folderTest + 'interpolatedSolutions_coarseP1.hd5'
nameInterpolationBasis = folderBasis + 'interpolatedSolutions_{0}_{1}.hd5'
nameTau = folderTest + 'tau_{0}_{1}_{2}.hd5'
EpsFlucPrefix = folderTestDATA + 'EpsList_periodic_offset0_{0}.txt'

# dotProduct = lambda u,v, dx : assemble(inner(u,v)*dx)
dotProduct = lambda u,v, m : assemble(inner(grad(u),grad(v))*m.dx)
# dotProduct = lambda u,v, dx : assemble(inner(grad(u),grad(v))*dx)
# dotProduct = lambda u,v, dx : assemble(inner(u,v)*dx) + assemble(inner(grad(u),grad(v))*dx) 
# dotProduct = lambda u,v, dx : np.dot(u.vector()[:],v.vector()[:]) 

ns = 5
Nmax = 156
Nblocks = 1

#lite, nx = 150, CG 3
#lite2, nx = 100, CG 2
#nonlite, nx = 300, CG 3
# meshRef = refine(fela.EnrichedMesh(nameMeshRef))
# Vref = VectorFunctionSpace(meshRef,"CG", 1)
nx = 100
meshRef = RectangleMesh(Point(1.0/3., 1./3.), Point(2./3., 2./3.), nx, nx, diagonal='crossed')
# W0S = FunctionSpace(mesh0, 'CG', 4)
Vref = VectorFunctionSpace(meshRef,"CG", 2)

dxRef = Measure('dx', meshRef) 
# dsRef = Measure('ds', meshRef) 


EpsList_name = 'EpsList_0.txt'
EpsFluc = np.zeros((ns,4))
for i in range(ns):
    EpsFluc[i,:] = np.loadtxt(EpsFlucPrefix.format(i))
    
np.savetxt(EpsList_name , EpsFluc)


# stress_name = 'SigmaList0.txt'
# sigma = np.zeros((ns,4))
# sigmaPrefix = folderTestDATA + 'sigmaL_{0}_offset{1}_{2}.txt'
# for i in range(ns):
#     sigma[i,:] = np.loadtxt(EpsFlucPrefix.format(i))
    
# np.savetxt(EpsList_name , EpsFluc)


# Exporting interpolation matrix to ease other calculations
# Isol , f = myhd.zeros_openFile(nameInterpolation, (ns, Vref.dim()), 'Isol')
# gdb.interpolationSolutions(Isol,Vref,Nmax, radFile, nameSol)
# f.close()


# #  ================  Extracting Alphas ============================================
# Wbasis, fw = myhd.loadhd5_openFile(nameWbasis.format(formulationLabel2,simul_id,EpsDirection), 'Wbasis')
# Isol, fIsol = myhd.loadhd5_openFile(nameInterpolation.format(simul_id,EpsDirection),'Isol')
# Ylist, f = myhd.zeros_openFile(nameYlist.format(formulationLabel2,simul_id,EpsDirection), (ns,Nmax) , 'Ylist')
# gdb.getAlphas(Ylist,Wbasis,Isol,ns,Nmax, radFile, nameSol, dotProduct, Vref, dxRef) 
# f.close()
# fIsol.close()
# fw.close()

# Computing basis for stress
E1 = 10.0
nu = 0.3
contrast = 0.1 #inverse then in generation
ns = 5
Nmax = 40
Wbasis, fw = myhd.loadhd5_openFile(nameWbasis.format(formulationLabel3,simul_id,EpsDirection), 'Wbasis')
tau, f = myhd.zeros_openFile(nameTau.format(formulationLabel2, 'corrected', EpsDirection), [(ns,Nmax,3),(ns,3)]  , ['tau', 'tau_0'])
# gdb.getStressBasis_Vrefbased(tau,Wbasis, ns, Nmax, EpsList_name, nameMeshPrefix, Vref, [E1,nu, contrast], EpsDirection)
gdb.getStressBasis(tau,Wbasis, ns, Nmax, EpsList_name, nameMeshPrefix, Vref, [E1,nu, contrast], EpsDirection, 'solvePDE_BC')
fw.close()
f.close()
