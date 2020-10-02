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
import fenicsUtils as feut 


simul_id = 3
EpsDirection = 0

print("starting with", simul_id, EpsDirection)
DATAfolder = "/Users/felipefr/EPFL/newDLPDES/DATA/"
folderBasis = DATAfolder + "deepBoundary/training3Nets/definitiveBasis/"
folderTest = "./"
folder_ = "/Users/felipefr/EPFL/newDLPDES/DATA/deepBoundary/data{0}/"
folder = folder_.format(simul_id)
radFile = folder + "RVE_POD_reduced_{0}.{1}"
nameMeshPrefix = folder + "RVE_POD_reduced_{0}.{1}"
nameWbasis = folderBasis + 'Wbasis_{0}_{1}_{2}.hd5'
nameYlist = folderTest + 'Y_train_{0}.hd5'
nameInterpolation = folderBasis + 'interpolatedSolutions_lite2_3_0.hd5'
nameTau = folderTest + 'tau_train_{0}.hd5'
nameEpsFluc = folder + 'EpsList_0.txt'


formulationLabel = "L2bnd"
formulationLabel3 = "L2bnd_converted"

dotProduct = lambda u,v, ds : assemble(inner(u,v)*ds)
# dotProduct = lambda u,v, m : assemble(inner(grad(u),grad(v))*m.dx)
# dotProduct = lambda u,v, dx : assemble(inner(grad(u),grad(v))*dx)
# dotProduct = lambda u,v, dx : assemble(inner(u,v)*dx) + assemble(inner(grad(u),grad(v))*dx) 
# dotProduct = lambda u,v, dx : np.dot(u.vector()[:],v.vector()[:]) 

ns = 20
Nmax = 156

nx = 100
meshRef = RectangleMesh(Point(1.0/3., 1./3.), Point(2./3., 2./3.), nx, nx, diagonal='crossed')
Vref = VectorFunctionSpace(meshRef,"CG", 2)

dxRef = Measure('dx', meshRef) 
dsRef = Measure('ds', meshRef) 

# Exporting interpolation matrix to ease other calculations
# Isol , f = myhd.zeros_openFile(nameInterpolation, (ns, Vref.dim()), 'Isol')
# gdb.interpolationSolutions(Isol,Vref,ns, radFile, nameSol)
# f.close()

# #  ================  Extracting Alphas ============================================
# Wbasis, fw = myhd.loadhd5_openFile(nameWbasis.format(formulationLabel3,simul_id,EpsDirection), 'Wbasis')
# Isol, fIsol = myhd.loadhd5_openFile(nameInterpolation,'Isol')
# Ylist, f = myhd.zeros_openFile(nameYlist.format(formulationLabel), (ns,Nmax) , 'Ylist')
# gdb.getAlphas(Ylist,Wbasis,Isol,ns,Nmax, radFile, dotProduct, Vref, dsRef) 
# f.close()
# fIsol.close()
# fw.close()

# Computing basis for stress
E1 = 10.0
nu = 0.3
contrast = 0.1 #inverse then in generation
Nmax = 156
Wbasis, fw = myhd.loadhd5_openFile(nameWbasis.format(formulationLabel3,simul_id,EpsDirection), 'Wbasis')
tau, f = myhd.zeros_openFile(nameTau.format(formulationLabel), [(ns,Nmax,3),(ns,3),(ns,3)]  , ['tau', 'tau_0', 'tau_0_fluc'])
# gdb.getStressBasis_Vrefbased(tau,Wbasis, ns, Nmax, "", nameMeshPrefix, Vref, [E1,nu, contrast], EpsDirection)
gdb.getStressBasis(tau,Wbasis, ns, Nmax, nameEpsFluc , nameMeshPrefix, Vref, [E1,nu, contrast], EpsDirection, 'solvePDE_BC')
fw.close()
f.close()

