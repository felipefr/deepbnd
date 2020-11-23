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

folder = ["/Users", "/home"][0] + "/felipefr/switchdrive/scratch/deepBoundary/smartGeneration/LHS_maxmin_full/"
nameSnaps = folder + 'snapshots_1.h5'
nameC = folder + 'Cnew.h5'
nameMeshRefBnd = 'boundaryMesh.xdmf'
nameWbasis = folder + 'Wbasis_new.h5'
nameYlist = folder + 'Y.h5'
nameTau = folder + 'tau2.h5'
nameEllipseData = folder + 'ellipseData_1.h5'

dotProduct = lambda u,v, dx : assemble(inner(u,v)*ds)

# os.system('rm ' + nameEllipseData.format('all'))
# myhd.merge([nameEllipseData.format(i) for i in range(1,11)], nameEllipseData.format('all'), 
#            InputLabels = ['ellipseData'], OutputLabels = ['ellipseData'], axis = 0, mode = 'w-')

# os.system('rm ' + nameSnaps.format('all'))
# myhd.merge([nameSnaps.format(i) for i in range(1,11)], nameSnaps.format('all'), 
#            InputLabels = ['solutions', 'a','B', 'sigma'], OutputLabels = ['solutions', 'a','B', 'sigma'], axis = 0, mode = 'w-')


Mref = meut.EnrichedMesh(nameMeshRefBnd)
Vref = VectorFunctionSpace(Mref,"CG", 1)

dxRef = Measure('dx', Mref) 
dsRef = Measure('ds', Mref) 

ns = 10000
Nmax = 160

# Translating solution
Isol, fIsol = myhd.loadhd5_openFile(nameSnaps,['solutions','a','B'], mode = 'a')
Isol_full , Isol_a, Isol_B = Isol
Isol_trans = Isol_full[:,:]
usol = Function(Vref)
normal = FacetNormal(Mref)
for i in range(ns):
    print('translating ', i)
    usol.vector().set_local(Isol_full[i,:])
    T = feut.affineTransformationExpession(Isol_a[i,:],Isol_B[i,:,:], Mref)
    Isol_trans[i,:] = Isol_trans[i,:] + interpolate(T,Vref).vector().get_local()[:] 
    
myhd.addDataset(fIsol,Isol_trans, 'solutions_trans')
fIsol.close()

# Computing Correlation Matrix 
os.system('rm ' + nameC)
C, fC = myhd.zeros_openFile(nameC, (ns, ns), 'C')
Isol, fIsol = myhd.loadhd5_openFile(nameSnaps,'solutions_trans', mode = 'r')
gdb.getCorrelation_fromInterpolationMatricial(C,ns, Isol, dotProduct, Vref, dsRef)
# C = Isol[:,:160]@Isol[:,:160].T
fC.close()
fIsol.close()

# Computing basis 
# os.system('rm ' + nameWbasis)
# C, fC = myhd.loadhd5_openFile(nameC,'C')
# Isol, fIsol = myhd.loadhd5_openFile(nameSnaps,'solutions_trans', mode = 'r')
# Wbasis, f = myhd.zeros_openFile(nameWbasis, (Nmax,Vref.dim()), 'Wbasis')
# gdb.computingBasis(Wbasis,C,Isol,Nmax)
# f.close()
# fC.close()
# fIsol.close()


#  ================  Extracting Alphas ============================================
# os.system('rm ' + nameYlist)
# Wbasis, fw = myhd.loadhd5_openFile(nameWbasis, 'Wbasis')
# Isol, fIsol = myhd.loadhd5_openFile(nameSnaps,'solutions_trans')
# Ylist, f = myhd.zeros_openFile(nameYlist, (ns,Nmax) , 'Ylist')
# gdb.getAlphas(Ylist,Wbasis,Isol,ns,Nmax, dotProduct, Vref, dsRef) 
# f.close()
# fIsol.close()
# fw.close()

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