import sys, os
import numpy as np
sys.path.insert(0, '../../utils/')

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
folderBasis = "./definitiveBasis/"
formulationLabel = "L2bnd"
formulationLabel2 = "L2bnd_original_solverPDE"
formulationLabel3 = "L2bnd_original"
folder_ = "/Users/felipefr/EPFL/newDLPDES/DATA/deepBoundary/data{0}/"
folder = folder_.format(simul_id)
radFile = folder + "RVE_POD_reduced_{0}.{1}"
nameSol = folder + 'RVE_POD_solution_red_{0}_' + str(EpsDirection) + '.{1}'
nameC = folderBasis + 'C_{0}_{1}_{2}.hd5'
nameMeshPrefix = folder + "RVE_POD_reduced_{0}.{1}"
nameMeshRef = nameMeshPrefix.format(0,'xml') # it'll be generated
nameWbasis = folderBasis + 'Wbasis_{0}_{1}_{2}.hd5'
nameYlist = folderBasis + 'Y_{0}_{1}_{2}.hd5'
nameInterpolation = folderBasis + 'interpolatedSolutions_{0}_{1}.hd5'
nameTau = folderBasis + 'tau_{0}_{1}_{2}.hd5'
EpsFlucPrefix = folder + 'EpsList_{0}.txt'

dotProduct = lambda u,v, dx : assemble(inner(u,v)*ds)
# dotProduct = lambda u,v, m : assemble(inner(grad(u),grad(v))*m.dx)
# dotProduct = lambda u,v, dx : assemble(inner(grad(u),grad(v))*dx)
# dotProduct = lambda u,v, dx : assemble(inner(u,v)*dx) + assemble(inner(grad(u),grad(v))*dx) 
# dotProduct = lambda u,v, dx : np.dot(u.vector()[:],v.vector()[:]) 

ns = 1000
Nmax = 156
Nblocks = 1

#lite, nx = 150, CG 3
#lite2, nx = 100, CG 2
#nonlite, nx = 300, CG 3
meshRef = refine(fela.EnrichedMesh(nameMeshRef))
Vref = VectorFunctionSpace(meshRef,"CG", 1)
# nx = 100
# meshRef = RectangleMesh(Point(1.0/3., 1./3.), Point(2./3., 2./3.), nx, nx, diagonal='crossed')
# W0S = FunctionSpace(mesh0, 'CG', 4)
# Vref = VectorFunctionSpace(meshRef,"CG", 2)

# dxRef = Measure('dx', meshRef) 
dsRef = Measure('ds', meshRef) 


# Exporting interpolation matrix to ease other calculations
# Isol , f = myhd.zeros_openFile(nameInterpolation.format(simul_id,EpsDirection), (ns, Vref.dim()), 'Isol')
# gdb.interpolationSolutions(Isol,Vref,Nmax, radFile, nameSol)
# f.close()
# ##myhd.savehd5(nameInterpolation.format(simul_id,EpsDirection),Isol,'Isol')

# Computing Correlation Matrix 
# C, fC = myhd.zeros_openFile(nameC.format(formulationLabel3, simul_id,EpsDirection), (ns, ns), 'C')
# Isol, fIsol = myhd.loadhd5_openFile(nameInterpolation.format(simul_id,EpsDirection),'Isol')
# gdb.getCorrelation_fromInterpolation(C,ns, Isol, dotProduct, radFile, nameSol, Vref, dxRef)
# fC.close()
# fIsol.close()

# Computing basis 
C, fC = myhd.loadhd5_openFile(nameC.format(formulationLabel3,simul_id,EpsDirection),'C')
Isol, fIsol = myhd.loadhd5_openFile(nameInterpolation.format(simul_id,EpsDirection),'Isol')
Wbasis, f = myhd.zeros_openFile(nameWbasis.format(formulationLabel2,simul_id,EpsDirection), (Nmax,Isol.shape[1]), 'Wbasis')
gdb.computingBasis(Wbasis,C,Isol,Nmax,radFile, nameSol)
f.close()
fC.close()
fIsol.close()


# #  ================  Extracting Alphas ============================================
# Wbasis, fw = myhd.loadhd5_openFile(nameWbasis.format(formulationLabel2,simul_id,EpsDirection), 'Wbasis')
# Isol, fIsol = myhd.loadhd5_openFile(nameInterpolation.format(simul_id,EpsDirection),'Isol')
# Ylist, f = myhd.zeros_openFile(nameYlist.format(formulationLabel2,simul_id,EpsDirection), (ns,Nmax) , 'Ylist')
# gdb.getAlphas(Ylist,Wbasis,Isol,ns,Nmax, radFile, nameSol, dotProduct, Vref, dxRef) 
# f.close()
# fIsol.close()
# fw.close()

# Computing basis for stress
# E1 = 10.0
# nu = 0.3
# contrast = 0.1 #inverse then in generation
# ns = 100
# Nmax = 156
# Wbasis, fw = myhd.loadhd5_openFile(nameWbasis.format(formulationLabel3,simul_id,EpsDirection), 'Wbasis')
# tau, f = myhd.zeros_openFile(nameTau.format(formulationLabel2, simul_id, EpsDirection), [(ns,Nmax,3),(ns,3)]  , ['tau', 'tau_0'])
# # gdb.getStressBasis_Vrefbased(tau,Wbasis, ns, Nmax, EpsFlucPrefix, nameMeshPrefix, Vref, [E1,nu, contrast], EpsDirection)
# gdb.getStressBasis(tau,Wbasis, ns, Nmax, EpsFlucPrefix, nameMeshPrefix, Vref, [E1,nu, contrast], EpsDirection,"solvePDE_BC")
# fw.close()
# f.close()


# scaling
# alpha= 1.0/np.sqrt(10.0)
# import h5py

# fw = h5py.File(nameWbasis.format(formulationLabel2,simul_id,EpsDirection),'r+')
# Wbasis = fw['Wbasis']
# Wbasis[:,:] = alpha*Wbasis[:,:]
# fw.close()

# ftau = h5py.File(nameTau.format(formulationLabel2,simul_id,EpsDirection),'r+')
# tau = ftau['tau']
# tau[:,:,:] = alpha*tau[:,:,:]
# ftau.close()

# fy = h5py.File(nameYlist.format(formulationLabel2,simul_id,EpsDirection),'r+')
# Ylist = fy['Ylist']
# Ylist[:,:] = alpha*Ylist[:,:]
# fy.close()

# Converting to hd5