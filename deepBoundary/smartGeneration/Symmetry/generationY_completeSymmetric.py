import sys, os
import numpy as np
sys.path.insert(0, '../../../utils/')
# sys.path.insert(0, '../training3Nets/')

import fenicsWrapperElasticity as fela
import generation_deepBoundary_lib as gdb
from dolfin import *
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import myHDF5 as myhd
import meshUtils as meut
import fenicsUtils as feut

f = open("../../../../rootDataPath.txt")
rootData = f.read()[:-1]
f.close()

folder = rootData + "/deepBoundary/smartGeneration/LHS_p4_fullSymmetric/"
folderBasis = rootData + "/deepBoundary/smartGeneration/LHS_p4_fullSymmetric/"

nameSnaps_simple = folder + 'snapshots_simple.h5'
nameSnaps = folder + 'snapshots_all.h5'
nameC = folderBasis + 'C.h5'
nameMeshRefBnd = 'boundaryMesh.xdmf'
nameWbasis = folderBasis + 'Wbasis.h5'
nameYlist = folder + 'Y.h5'
nameTau = folderBasis + 'tau.h5'
nameEllipseData = folder + 'ellipseData.h5'

dotProduct = lambda u,v, dx : assemble(inner(u,v)*ds)

Mref = meut.EnrichedMesh(nameMeshRefBnd)
Vref = VectorFunctionSpace(Mref,"CG", 1)

dxRef = Measure('dx', Mref) 
dsRef = Measure('ds', Mref) 

ns = 10240
Nmax = 160

#Isol, fIsol = myhd.loadhd5_openFile(nameSnaps_simple,'solutions_trans', mode = 'r')
#os.system('rm ' + nameSnaps)
#Isol_all, fIsol_all = myhd.zeros_openFile(nameSnaps , (4*ns, Vref.dim()), 'solutions_trans')
#
#w0 = Function(Vref)
#g1 = Expression(('-x[0]','x[1]'), degree = 1)
#g2 = Expression(('x[0]','-x[1]'), degree = 1)
#g3 = Expression(('-x[0]','-x[1]'), degree = 1)
#
#for i in range(ns):
#    print('symmetrising i = ', i)
#    w0.vector().set_local(Isol[i,:])
#    w1 = interpolate(feut.myfog(w0,g1),Vref)
#    w2 = interpolate(feut.myfog(w0,g2),Vref)
#    w3 = interpolate(feut.myfog(w0,g3),Vref)
#    
#    Isol_all[i,:] = w0.vector().get_local()
#    Isol_all[ns + i,:] = w1.vector().get_local()
#    Isol_all[2*ns + i,:] = w2.vector().get_local()
#    Isol_all[3*ns + i,:] = w3.vector().get_local()
# 
#fIsol.close()
#fIsol_all.close()

 # Computing Correlation Matrix 
#os.system('rm ' + nameC)
#C, fC = myhd.zeros_openFile(nameC, (4*ns, 4*ns), 'C')
#Isol, fIsol = myhd.loadhd5_openFile(nameSnaps,'solutions_trans', mode = 'r')
#gdb.getCorrelation_fromInterpolationMatricial(C,4*ns, Isol, dotProduct, Vref, dsRef)
#C = Isol[:,:160]@Isol[:,:160].T
#fC.close()
#fIsol.close()

# # Computing basis 
os.system('rm ' + nameWbasis)
C, fC = myhd.loadhd5_openFile(nameC,'C')
Isol, fIsol = myhd.loadhd5_openFile(nameSnaps,'solutions_trans', mode = 'r')
Wbasis, f = myhd.zeros_openFile(nameWbasis, (Nmax,Vref.dim()), 'Wbasis')
sig, U = gdb.computingBasis(Wbasis,C,Isol,Nmax)
myhd.savehd5(folder + 'eigens.hd5', [sig,U],['eigenvalues','eigenvectors'], mode = 'w-')
f.close()
fC.close()
fIsol.close()

# Recompute Basis Symmetries
# os.system('rm ' + nameWbasisT1)
# os.system('rm ' + nameWbasisT2)
# os.system('rm ' + nameWbasisT3)
# Wbasis, fw = myhd.loadhd5_openFile(nameWbasis, 'Wbasis')
# WbasisT1, f1 = myhd.zeros_openFile(nameWbasisT1, (Nmax,Vref.dim()), 'Wbasis')
# WbasisT2, f2 = myhd.zeros_openFile(nameWbasisT2, (Nmax,Vref.dim()), 'Wbasis')
# WbasisT3, f3 = myhd.zeros_openFile(nameWbasisT3, (Nmax,Vref.dim()), 'Wbasis')


# #  ================  Extracting Alphas ============================================
# os.system('rm ' + nameYlistT3)
# Wbasis, fw = myhd.loadhd5_openFile(nameWbasisT3, 'Wbasis')
# Isol = myhd.loadhd5(nameSnaps,'solutions_trans')
# Ylist, f = myhd.zeros_openFile(nameYlistT3, (ns,Nmax) , 'Ylist')
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