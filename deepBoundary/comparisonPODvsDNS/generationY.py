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
import fenicsMultiscale as fmts

DATAfolder = "/Users/felipefr/EPFL/newDLPDES/DATA/"
folderBasis = DATAfolder + "deepBoundary/training3Nets/definitiveBasis/"
folderTest = "./"
folderTestDATA = DATAfolder + "deepBoundary/comparisonPODvsDNS/Per/"
folderTestMeshes = DATAfolder + "deepBoundary/comparisonPODvsDNS/meshes/"
radMesh = folderTestMeshes + "RVE_POD_reduced_{0}.{1}"
nameSol = folderTestDATA + 'RVE_POD_solRed_periodic_offset2_{0}.{1}'
nameWbasis = folderBasis + 'Wbasis_{0}_3_0.hd5'
nameYlist = folderTest + 'Y_{0}.hd5'
nameInterpolation = folderTest + 'interpolatedSolutions_Per.hd5'
nameTau = folderTest + 'tau_{0}.hd5'
nameEpsFluc = folderTestDATA + 'EpsList_periodic.hd5'

opForm = 0
formulationLabel = ["L2bnd_noOrth","H10_noOrth_VL"][opForm]
formulationLabel3 = ["L2bnd_converted", "H10_lite2_correction"][opForm]
dotProduct = [lambda u,v, ds : assemble(inner(u,v)*ds) , lambda u,v, dx : assemble(inner(grad(u),grad(v))*dx)][opForm]

ns = 20
Nmax = 156

nx = 100
meshRef = RectangleMesh(Point(1.0/3., 1./3.), Point(2./3., 2./3.), nx, nx, diagonal='crossed')
Vref = VectorFunctionSpace(meshRef,"CG", 2)

dsRef = Measure('ds', meshRef) 
dxRef = Measure('dx', meshRef) 
dm = [dsRef,dxRef][opForm]

# Exporting interpolation matrix to ease other calculations
# Isol , f = myhd.zeros_openFile(nameInterpolation, (ns, Vref.dim()), 'Isol')
# gdb.interpolationSolutions(Isol,Vref,ns, radMesh , nameSol)
# f.close()

# #  ================  Extracting Alphas ============================================
# Wbasis, fw = myhd.loadhd5_openFile(nameWbasis.format(formulationLabel3), 'Wbasis')
# Isol, fIsol = myhd.loadhd5_openFile(nameInterpolation,'Isol')
# Ylist, f = myhd.zeros_openFile(nameYlist.format(formulationLabel), (ns,Nmax) , 'Ylist')
# gdb.getAlphas(Ylist,Wbasis,Isol,ns,Nmax, radMesh, dotProduct, Vref, dm) 
# f.close()
# fIsol.close()
# fw.close()

# Computing basis for stress
E1 = 10.0
nu = 0.3
contrast = 0.1 #inverse then in generation
Nmax = 156
Wbasis, fw = myhd.loadhd5_openFile(nameWbasis.format(formulationLabel3), 'Wbasis')
tau, f = myhd.zeros_openFile(nameTau.format(formulationLabel), [(ns,Nmax,3),(ns,3),(ns,3)]  , ['tau', 'tau_0', 'tau_0_fluc'])
# gdb.getStressBasis_Vrefbased(tau,Wbasis, ns, Nmax, nameEpsFluc, radMesh, Vref, [E1,nu, contrast], EpsDirection = 0 )
# gdb.getStressBasis(tau,Wbasis, ns, Nmax, nameEpsFluc , radMesh, Vref, [E1,nu, contrast], EpsDirection = 0 , op = 'periodic')
gdb.getStressBasis_generic(tau, Wbasis, Wbasis, ns, Nmax,  nameEpsFluc , radMesh, Vref, [E1,nu, contrast], EpsDirection = 0, V0 = 'VL', Orth = False)
fw.close()
f.close()



# WbasisNew, fwnew = myhd.zeros_openFile(nameWbasis.format('L2bnd_converted',simul_id,EpsDirection), (Wbasis.shape[0],Vref.dim())  , 'Wbasis')
# gdb.reinterpolateWbasis(WbasisNew, Vref, Wbasis, Vref0)
# fwnew.close()
# fw.close()

# Building the antiperiodic Basis
# WbasisNew, fwnew = myhd.zeros_openFile(nameWbasis.format('L2bnd_antiperiodic'), (Wbasis.shape[0],Vref.dim())  , 'Wbasis')
# basis = Function(Vref)
# for i in range(Nmax):
#     print("computing basis ", i)
#     basis.vector().set_local(Wbasis[i,:])
#     WbasisNew[i,:] = fmts.getAntiperiodic(basis).vector()[:] 
    
# fwnew.close()
# fw.close()
