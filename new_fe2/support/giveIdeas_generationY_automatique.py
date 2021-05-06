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
import meshUtils as meut
import fenicsUtils as feut

folder = "/Users/felipefr/EPFL/newDLPDES/DATA/deepBoundary/generateDataset/"
nameSol = folder + 'axial/snapshots.h5'
nameC = folder + 'axial/C.h5'
nameMeshRef = folder + "meshRef.xdmf"
nameMeshRefBnd = 'boundaryMesh.xdmf'
nameWbasis = folder + 'axial/Wbasis.h5'
nameYlist = folder + 'axial/Y.h5'
nameTau = folder + 'axial/tau2.h5'
nameSnaps = folder + 'axial/snapshots_all.h5'
nameEllipseData = folder + 'ellipseData_all.h5'
nameEpsFluc = 'EpsList.h5'
radMesh = folder + "meshesReduced/mesh_{0}.xdmf"

# DATAfolder = "/Users/felipefr/EPFL/newDLPDES/DATA/"
# folderBasis = DATAfolder + "deepBoundary/training3Nets/definitiveBasis/"
# folderTest = "./"
# folderTestDATA = DATAfolder + "deepBoundary/comparisonPODvsDNS/Per/"
# folderTestMeshes = DATAfolder + "deepBoundary/comparisonPODvsDNS/meshes/"
# radMesh = folderTestMeshes + "RVE_POD_reduced_{0}.{1}"
# nameSol = folderTestDATA + 'RVE_POD_solRed_periodic_offset2_{0}.{1}'
# nameWbasis = folderBasis + 'Wbasis_{0}_3_0.hd5'
# nameYlist = folderTest + 'Y_{0}.hd5'
# nameInterpolation = folderTest + 'interpolatedSolutions_Per.hd5'
# nameTau = folderTest + 'tau_{0}.hd5'


dotProduct = lambda u,v, ds : assemble(inner(u,v)*ds)

ns = 20
Nmax = 100

Mref = meut.EnrichedMesh(nameMeshRefBnd)
Vref = VectorFunctionSpace(Mref,"CG", 1)

dxRef = Measure('dx', Mref) 
dsRef = Measure('ds', Mref) 
dm = [dsRef,dxRef][0]


# Exporting EpsFluc
# os.system('rm ' + nameEpsFluc)
# EpsFluc, feps = myhd.zeros_openFile(nameEpsFluc, (ns,4) , 'EpsList')
# Isol, fIsol = myhd.loadhd5_openFile(nameSnaps,['solutions','a'], mode = 'r')
# Isol_full , Isol_a = Isol
# Isol_trans = Isol_full[:,:]
# usol = Function(Vref)
# normal = FacetNormal(Mref)
# for i in range(ns):
#     usol.vector().set_local(Isol_full[i,:])
#     B = feut.Integral(outer(usol,normal), dsRef, shape = (2,2))/4.0
#     EpsFluc[i,:] = B.flatten()

# fIsol.close()
# feps.close()

# Creating reduced meshes
# ellipseData, fellipse = myhd.loadhd5_openFile(filename = nameEllipseData, label = 'ellipseData')
# for i in range(ns): 
#     meshGMSH = meut.ellipseMesh2(ellipseData[i,:4,:], x0 = -1.0, y0 = -1.0, Lx = 2.0 , Ly = 2.0 , lcar = 0.1)
#     meshGMSH.setTransfiniteBoundary(21)
#     meshGMSH.setNameMesh(radMesh.format(i))
#     mesh = meshGMSH.getEnrichedMesh() 

# fellipse.close()

LevelsSpaces = ['L']
LevelsExtOrth = [1] # yes or no (not bool because it may change)
LevelsRBbasis = ['L2bnd']
LevelsMetric = ['V']

cases = {}
for s,e,r,m  in product(LevelsSpaces, LevelsExtOrth, LevelsRBbasis, LevelsMetric):
    c = gdb.SimulationCase(s,e,r,m)
    if(c.checkException()):
        cases[c.getLabel()] = c

case = c

Ns = 20
E1 = 10.0
nu = 0.3
contrast = 0.1 #inverse then in generation
os.system('rm base.hd5')
RBs = gdb.RBsimul('base.hd5',  [E1,nu, contrast], nameEpsFluc, Vref, radMesh, Nmax, EpsDirection = 0)   

RBs.registerFile(nameSnaps, '', 'solutions_trans')
RBs.registerFile(nameWbasis , '', 'Wbasis')

# RBs.closeAllFiles() 
# RBs.resetSimulationCase(gdb.SimulationCase('P',1,'L2bnd','V'))

for i in range(Ns):
    RBs.resetSimulationCase(c) # all rest dummy
    RBs.computeU0s(i)
    
for i in range(Ns):
    RBs.resetSimulationCase(c) # all rest dummy
    RBs.computeEtas(i)
    
for i in range(Ns):
    RBs.resetSimulationCase(c) # all rest dummy
    RBs.computeStressBasis0(i)
    
for i in range(Ns):
    RBs.resetSimulationCase(c) # all rest dummy
    RBs.computeAdjointBasis(i)

for i in range(Ns):
    RBs.resetSimulationCase(c) # all rest dummy
    RBs.computeStressBasisRB(i)

RBs.closeAllFiles()



# X, f = myhd.loadhd5_openFile('base.hd5', 'U0/{0}/{1}'.format('L',0))


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