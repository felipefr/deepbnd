import sys, os
import numpy as np
sys.path.insert(0, '../../utils/')
# sys.path.insert(0, '../training3Nets/')

import fenicsWrapperElasticity as fela
import generation_deepBoundary_lib as gdb
from dolfin import *
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import myHDF5 as myhd
import meshUtils as meut
import fenicsUtils as feut

folder = ["/Users", "/home"][1] + "/felipefr/switchdrive/scratch/deepBoundary/smartGeneration/validation_and_test/"
folderBasis = ["/Users", "/home"][1] + "/felipefr/switchdrive/scratch/deepBoundary/smartGeneration/LHS_frozen_p4_volFraction/"
nameSnaps = folder + 'snapshots_{0}_copy.h5'
nameC = folderBasis + 'Cnew.h5'
nameMeshRefBnd = 'boundaryMesh.xdmf'
nameWbasis = folderBasis + 'Wbasis_new.h5'
nameYlist = folder + 'Y_validation_p4.h5'
nameTau = folderBasis + 'tau2.h5'
nameEllipseData = folder + 'ellipseData_validation.h5'

dotProduct = lambda u,v, dx : assemble(inner(u,v)*ds)

Mref = meut.EnrichedMesh(nameMeshRefBnd)
Vref = VectorFunctionSpace(Mref,"CG", 1)

dxRef = Measure('dx', Mref) 
dsRef = Measure('ds', Mref) 

# nperpartition = 2048
# os.system('rm ' + folder +  'snapshots_compound.h5')
# snapshots, fsnaps = myhd.zeros_openFile(filename = folder +  'snapshots_compound.h5',  
#                                         shape = [(nperpartition,Vref.dim()),(nperpartition,3),(nperpartition,2),
#                                                   (nperpartition,2,2)], label = ['solutions','sigma','a','B'], mode = 'w-')

# snap_solutions, snap_sigmas, snap_a, snap_B = snapshots

# snapshots1 = myhd.loadhd5( folder +  'snapshots_0.h5', label = ['solutions','sigma','a','B'])
# snapshots2 = myhd.loadhd5( folder +  'snapshots_1_complement.h5', label = ['solutions','sigma','a','B'])


# snap_solutions1, snap_sigmas1, snap_a1, snap_B1 = snapshots1
# snap_solutions2, snap_sigmas2, snap_a2, snap_B2 = snapshots2

# threshold = 1527
# snap_solutions[:threshold,:] = snap_solutions1[:threshold,:] 
# snap_solutions[threshold:,:] = snap_solutions2[1:474,:] 

# snap_sigmas[:threshold,:] = snap_sigmas1[:threshold,:] 
# snap_sigmas[threshold:,:] = snap_sigmas2[1:474,:] 

# snap_a[:threshold,:] = snap_a1[:threshold,:] 
# snap_a[threshold:,:] = snap_a2[1:474,:] 

# snap_B[:threshold,:] = snap_B1[:threshold,:] 
# snap_B[threshold:,:] = snap_B2[1:474,:] 

# fsnaps.close()

# os.system('rm ' + nameEllipseData.format('all'))
# myhd.merge([nameEllipseData.format(i) for i in range(1,11)], nameEllipseData.format('all'), 
#             InputLabels = ['ellipseData'], OutputLabels = ['ellipseData'], axis = 0, mode = 'w-')

# os.system('rm ' + nameSnaps.format('test'))
# myhd.merge([nameSnaps.format(i) for i in range(2)], nameSnaps.format('all'), 
            # InputLabels = ['solutions', 'a','B', 'sigma'], OutputLabels = ['solutions', 'a','B', 'sigma'], axis = 0, mode = 'w-')


ns = 2000
Nmax = 160

# Translating solution
# Isol, fIsol = myhd.loadhd5_openFile(nameSnaps.format('validation'),['solutions','a','B'], mode = 'a')
# Isol_full , Isol_a, Isol_B = Isol
# Isol_trans = Isol_full[:,:]
# usol = Function(Vref)
# normal = FacetNormal(Mref)
# for i in range(ns):
#     print('translating ', i)
#     usol.vector().set_local(Isol_full[i,:])
#     T = feut.affineTransformationExpession(Isol_a[i,:],Isol_B[i,:,:], Mref)
#     Isol_trans[i,:] = Isol_trans[i,:] + interpolate(T,Vref).vector().get_local()[:] 
    
# myhd.addDataset(fIsol,Isol_trans, 'solutions_trans')
# fIsol.close()

# # Computing Correlation Matrix 
# os.system('rm ' + nameC)
# C, fC = myhd.zeros_openFile(nameC, (ns, ns), 'C')
# Isol, fIsol = myhd.loadhd5_openFile(nameSnaps.format('all'),'solutions_trans', mode = 'r')
# gdb.getCorrelation_fromInterpolationMatricial(C,ns, Isol, dotProduct, Vref, dsRef)
# # C = Isol[:,:160]@Isol[:,:160].T
# fC.close()
# fIsol.close()

# # Computing basis 
# os.system('rm ' + nameWbasis)
# C, fC = myhd.loadhd5_openFile(nameC,'C')
# Isol, fIsol = myhd.loadhd5_openFile(nameSnaps.format('all'),'solutions_trans', mode = 'r')
# Wbasis, f = myhd.zeros_openFile(nameWbasis, (Nmax,Vref.dim()), 'Wbasis')
# sig, U = gdb.computingBasis(Wbasis,C,Isol,Nmax)
# myhd.savehd5(folder + 'eigens.hd5', [sig,U],['eigenvalues','eigenvectors'], mode = 'w-')
# f.close()
# fC.close()
# fIsol.close()


# #  ================  Extracting Alphas ============================================
os.system('rm ' + nameYlist)
Wbasis, fw = myhd.loadhd5_openFile(nameWbasis, 'Wbasis')
Isol = myhd.loadhd5(nameSnaps.format('validation'),'solutions_trans')
Ylist, f = myhd.zeros_openFile(nameYlist, (ns,Nmax) , 'Ylist')
gdb.getAlphas(Ylist,Wbasis,Isol,ns,Nmax, dotProduct, Vref, dsRef) 
f.close()
# fIsol.close()
fw.close()

# Computing basis for stress
# E1 = 10.0
# nu = 0.3
# contrast = 0.1 #inverse then in generation
# ns = 20
# Nmax = 400
# os.system('rm ' + nameTau)
# Isol, fIsol = myhd.loadhd5_openFile(nameSnaps,'solutions', mode = 'r')
# EllipseData, fellipse = myhd.loadhd5_openFile(filename = nameEllipseData, label = 'ellipseData')
# Wbasis, fw = myhd.loadhd5_openFile(nameWbasis, 'Wbasis')
# tau, f = myhd.zeros_openFile(nameTau, [(ns,Nmax,3),(ns,3)]  , ['tau', 'tau_0'])
# gdb.getStressBasis_noMesh(tau,Wbasis, Isol, EllipseData[:ns,:,:], Nmax, Vref, [E1,nu, contrast], EpsDirection = 0)
# fw.close()
# f.close()
# fellipse.close()
# fIsol.close()