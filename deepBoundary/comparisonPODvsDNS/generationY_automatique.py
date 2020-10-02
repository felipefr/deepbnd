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
from itertools import product

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


# LevelsSpaces = ['M','P','L','T']
# LevelsExtOrth = [1,0] # yes or no (not bool because it may change)
# LevelsRBbasis = ['L2bnd','H10','L2']
# LevelsMetric = ['a','V']

LevelsSpaces = ['P','L']
LevelsExtOrth = [1,0] # yes or no (not bool because it may change)
LevelsRBbasis = ['L2bnd','H10']
LevelsMetric = ['V']


cases = {}
for s,e,r,m  in product(LevelsSpaces, LevelsExtOrth, LevelsRBbasis, LevelsMetric):
    c = gdb.SimulationCase(s,e,r,m)
    if(c.checkException()):
        cases[c.getLabel()] = c
        
Ns = 20
E1 = 10.0
nu = 0.3
contrast = 0.1 #inverse then in generation
# os.system('rm base.hd5')
RBs = gdb.RBsimul('base.hd5',  [E1,nu, contrast], nameEpsFluc, Vref, radMesh, Nmax, EpsDirection = 0)   

RBs.registerFile(nameInterpolation, '', 'Isol')
RBs.registerFile(nameWbasis.format('H10_lite2_correction'), 'H10', 'Wbasis')
RBs.registerFile(nameWbasis.format('L2bnd_converted'), 'L2bnd', 'Wbasis')
RBs.registerFile(nameWbasis.format('L2bnd_antiperiodic'), 'L2bnd_A', 'Wbasis')

# RBs.closeAllFiles() 
# RBs.resetSimulationCase(gdb.SimulationCase('P',1,'L2bnd','V'))

# for i,V0 in product(range(Ns),LevelsSpaces[0:3]):
#     RBs.resetSimulationCase(gdb.SimulationCase(V0,1,'L2bnd','V')) # all rest dummy
#     RBs.computeU0s(i)
    
# for i,V0, Vrb in product(range(Ns),LevelsSpaces[1:3],LevelsRBbasis[0:1]):
#     RBs.resetSimulationCase(gdb.SimulationCase(V0,1,Vrb,'V')) # all rest dummy
#     RBs.computeAdjointBasis(i)

# for i,V0 in product(range(Ns),LevelsSpaces):
#     RBs.resetSimulationCase(gdb.SimulationCase(V0,1,'L2bnd','V')) # all rest dummy
#     RBs.computeStressBasis0(i)

# for i,c in product(range(Ns),list(cases.values())):
#     RBs.resetSimulationCase(c) # all rest dummy
#     RBs.computeStressBasisRB(i)

# for i,c in product(range(1,Ns),list(cases.values())):
#     RBs.resetSimulationCase(c) # all rest dummy
#     RBs.computeEtas(i)
    
for i in range(Ns):
    RBs.resetSimulationCase(gdb.SimulationCase('P',1,'L2bnd','V')) # all rest dummy
    RBs.computeStressBasisRB(i)

for i in range(Ns):
    RBs.resetSimulationCase(gdb.SimulationCase('P',1,'L2bnd','V')) # all rest dummy
    RBs.computeEtas(i)

RBs.closeAllFiles()

# #  ================  Extracting Alphas ============================================
# Wbasis, fw = myhd.loadhd5_openFile(nameWbasis.format(formulationLabel3), 'Wbasis')
# Isol, fIsol = myhd.loadhd5_openFile(nameInterpolation,'Isol')
# Ylist, f = myhd.zeros_openFile(nameYlist.format(formulationLabel), (ns,Nmax) , 'Ylist')
# gdb.getAlphas(Ylist,Wbasis,Isol,ns,Nmax, radMesh, dotProduct, Vref, dm) 
# f.close()
# fIsol.close()
# fw.close()

# Computing basis for stress
# E1 = 10.0
# nu = 0.3
# contrast = 0.1 #inverse then in generation
# Nmax = 156
# Wbasis, fw = myhd.loadhd5_openFile(nameWbasis.format(formulationLabel3), 'Wbasis')
# tau, f = myhd.zeros_openFile(nameTau.format(formulationLabel), [(ns,Nmax,3),(ns,3),(ns,3)]  , ['tau', 'tau_0', 'tau_0_fluc'])
# # gdb.getStressBasis_Vrefbased(tau,Wbasis, ns, Nmax, nameEpsFluc, radMesh, Vref, [E1,nu, contrast], EpsDirection = 0 )
# # gdb.getStressBasis(tau,Wbasis, ns, Nmax, nameEpsFluc , radMesh, Vref, [E1,nu, contrast], EpsDirection = 0 , op = 'periodic')
# gdb.getStressBasis_generic(tau, Wbasis, Wbasis, ns, Nmax,  nameEpsFluc , radMesh, Vref, [E1,nu, contrast], EpsDirection = 0, V0 = 'VL', Orth = False)
# fw.close()
# f.close()



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
