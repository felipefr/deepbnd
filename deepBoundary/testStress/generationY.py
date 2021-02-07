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

f = open("../../../rootDataPath.txt")
rootData = f.read()[:-1]
f.close()

folder = rootData + "/deepBoundary/testStress/"
folderBasis = rootData + "/deepBoundary/smartGeneration/LHS_frozen_p4_volFraction/"
# folderBasis = rootData + "/deepBoundary/smartGeneration/LHS_p4_fullSymmetric/"

nameSnaps = folder + 'snapshots_17_{0}.h5'
nameMeshRefBnd = 'boundaryMesh.xdmf'
nameWbasis = folderBasis + 'Wbasis_new.h5'
nameYlist = folder + 'Y_{0}.h5'
nameTau = folder + 'tau_{0}.h5'
nameEllipseData = folder + 'ellipseData_17.h5'

dotProduct = lambda u,v, dx : assemble(inner(u,v)*ds)

Mref = meut.EnrichedMesh(nameMeshRefBnd)
Vref = VectorFunctionSpace(Mref,"CG", 1)

dxRef = Measure('dx', Mref) 
dsRef = Measure('ds', Mref) 

ns = 12
npar = 12
Nmax = 160
Npartitions = 3

op = int(sys.argv[1])
partition = int(sys.argv[2])

if(partition == -1):
    labelSnaps = 'all'
else:
    labelSnaps = str(partition)
    npar = int(ns/Npartitions)          
    n0 = partition*npar
    n1 = (partition+1)*npar
    
    
print('labelSnaps = ', labelSnaps)
    
if(op == 0):
    print('merging')
    os.system('rm ' + nameSnaps.format(labelSnaps))
    myhd.merge([nameSnaps.format(i) for i in range(Npartitions)], nameSnaps.format(labelSnaps), 
                InputLabels = ['solutions', 'a','B', 'sigma'], OutputLabels = ['solutions', 'a','B', 'sigma'], axis = 0, mode = 'w-')

            # InputLabels = ['solutions', 'a','B', 'sigma', 'sigmaT'], OutputLabels = ['solutions', 'a','B', 'sigma', 'sigmaT'], axis = 0, mode = 'w-')

# Translating solution
if(op == 1):
    Isol, fIsol = myhd.loadhd5_openFile(nameSnaps.format(labelSnaps),['solutions','a','B'], mode = 'a')
    Isol_full , Isol_a, Isol_B = Isol
    Isol_trans = Isol_full[:,:]
    usol = Function(Vref)
    normal = FacetNormal(Mref)
    for i in range(npar):
        print('translating ', i)
        usol.vector().set_local(Isol_full[i,:])
        T = feut.affineTransformationExpession(Isol_a[i,:],Isol_B[i,:,:], Mref)
        Isol_trans[i,:] = Isol_trans[i,:] + interpolate(T,Vref).vector().get_local()[:] 
        
    myhd.addDataset(fIsol,Isol_trans, 'solutions_trans')
    fIsol.close()

#  ================  Extracting Alphas ============================================
if(op == 2):
    os.system('rm ' + nameYlist.format(labelSnaps))
    Wbasis = myhd.loadhd5(nameWbasis, 'Wbasis')
    Isol = myhd.loadhd5(nameSnaps.format(labelSnaps),'solutions_trans')
    Ylist, f = myhd.zeros_openFile(nameYlist.format(labelSnaps), (npar,Nmax) , 'Ylist')
    gdb.getAlphas(Ylist,Wbasis,Isol,npar,Nmax, dotProduct, Vref, dsRef) 
    f.close()


# ======================= Computing basis for stress ==============================
if(op == 3):
    E1 = 10.0
    nu = 0.3
    contrast = 0.1 #inverse than in generation
    Nmax = 156
    os.system('rm ' + nameTau)
    Isol = myhd.loadhd5(nameSnaps.format(labelSnaps),'solutions_trans')
    EllipseData = myhd.loadhd5(filename = nameEllipseData, label = 'ellipseData')
    Wbasis = myhd.loadhd5(nameWbasis, 'Wbasis')
    tau, f = myhd.zeros_openFile(nameTau, [(ns,Nmax,3),(ns,3)]  , ['tau', 'tau_0'])
    # gdb.getStressBasis_noMesh(tau,Wbasis, Isol, EllipseData[:ns,:,:], Nmax, Vref, [E1,nu, contrast], EpsDirection = 1) # shear
    gdb.getStressBasis_noMesh_partitioned(tau,Wbasis, Isol, EllipseData[:ns,:,:], Nmax, Vref, [E1,nu, contrast],  1, n0, n1) # shear
    f.close()
