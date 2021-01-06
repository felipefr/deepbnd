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
import meshUtils as meut
import fenicsUtils as feut

folder = ["/Users", "/home"][1] + "/felipefr/EPFL/newDLPDEs/DATA/deepBoundary/RBsensibility/fullSampled/shear/"
nameC = folder + 'C.h5'
nameMeshRefBnd = 'boundaryMesh.xdmf'
nameWbasis = folder + 'Wbasis.h5'
nameSnaps = folder + 'snapshots_1.h5'

dotProduct = lambda u,v, dx : assemble(inner(u,v)*ds)

Mref = meut.EnrichedMesh(nameMeshRefBnd)
Vref = VectorFunctionSpace(Mref,"CG", 1)

dxRef = Measure('dx', Mref) 
dsRef = Measure('ds', Mref) 

ns = 1000
Nmax = 240

# Translating solution
# Isol, fIsol = myhd.loadhd5_openFile(nameSnaps,['solutions','a','B'], mode = 'r+')
# Isol_full , Isol_a, Isol_B = Isol
# Isol_trans = Isol_full[:,:]
# usol = Function(Vref)
# normal = FacetNormal(Mref)
# for i in range(ns):
#     print('translating ', i)
#     usol.vector().set_local(Isol_full[i,:])
#     T = feut.affineTransformationExpession(Isol_a[i,:],Isol_B[i,:], Mref)
#     Isol_trans[i,:] = Isol_trans[i,:] + interpolate(T,Vref).vector().get_local()[:] 
    
# myhd.addDataset(fIsol,Isol_trans, 'solutions_trans')
# fIsol.close()

# Computing Correlation Matrix 
# os.system('rm ' + nameC)
# C, fC = myhd.zeros_openFile(nameC, (ns, ns), 'C')
# Isol, fIsol = myhd.loadhd5_openFile(nameSnaps,'solutions_trans', mode = 'r')
# gdb.getCorrelation_fromInterpolation(C,ns, Isol, dotProduct, Vref, dsRef)
# fC.close()
# fIsol.close()

# Computing basis 
os.system('rm ' + nameWbasis)
C, fC = myhd.loadhd5_openFile(nameC,'C')
Isol, fIsol = myhd.loadhd5_openFile(nameSnaps,'solutions_trans', mode = 'r')
Wbasis, f = myhd.zeros_openFile(nameWbasis, (Nmax,Vref.dim()), 'Wbasis')
gdb.computingBasis(Wbasis,C,Isol,Nmax)
f.close()
fC.close()
fIsol.close()