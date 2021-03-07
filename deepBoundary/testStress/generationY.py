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
import symmetryLib as syml

# f = open("../../../rootDataPath.txt")
# rootData = f.read()[:-1]
# f.close()

folder = './models/dataset_hybrid/'
folderBasis = './models/dataset_hybrid/'

nameSnaps = folder + 'snapshots.h5'
nameMeshRefBnd = 'boundaryMesh.xdmf'
nameWbasis = folderBasis + 'Wbasis.h5'
nameYlist = folder + 'Y.h5'
nameXYlist = folder + 'XY.h5'
nameTau = folder + 'tau.h5'
nameEllipseData = folder + 'ellipseData.h5'

dotProduct = lambda u,v, dx : assemble(inner(u,v)*ds)

Mref = meut.EnrichedMesh(nameMeshRefBnd)
Vref = VectorFunctionSpace(Mref,"CG", 1)

dxRef = Measure('dx', Mref) 
dsRef = Measure('ds', Mref) 

ns = 40960
npar = ns
Nmax = 160
Npartitions = 10

# op = int(sys.argv[1])
# partition = int(sys.argv[2])

op = int(input('option'))
partition = int(input('partition'))

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
                InputLabels = ['solutions', 'a','B', 'sigma', 'sigmaTotal'], OutputLabels = ['solutions', 'a','B', 'sigma', 'sigmaTotal'], axis = 0, mode = 'w-')

if(op == 4):
    print('merging Y')
    os.system('rm ' + nameYlist.format(labelSnaps))
    myhd.merge([nameYlist.format(i) for i in range(Npartitions)], nameYlist.format(labelSnaps), 
                InputLabels = ['Ylist'], OutputLabels = ['Ylist'], axis = 0, mode = 'w-')

# Translating solution
if(op == 1):
    Isol, fIsol = myhd.loadhd5_openFile(nameSnaps.format(labelSnaps),['solutions','a','B'], mode = 'a')
    Isol_full , Isol_a, Isol_B = Isol
    Isol_trans = Isol_full[:,:]
    usol = Function(Vref)
    for i in range(npar):
        print('translating ', i)
        usol.vector().set_local(Isol_full[i,:])
        T = feut.affineTransformationExpession(Isol_a[i,:],Isol_B[i,:,:], Mref)
        Isol_trans[i,:] = Isol_trans[i,:] + interpolate(T,Vref).vector().get_local()[:] 
        
    myhd.addDataset(fIsol,Isol_trans, 'solutions_trans')
    fIsol.close()
    
# Mirroying solution
if(op == 6):
    Isol, f = myhd.loadhd5_openFile(nameSnaps.format(labelSnaps),'solutions_trans', mode = 'a')
    Transformations = [syml.T_MH,syml.T_MV,syml.T_MD]
    usol = Function(Vref)    

    for T, label_T in zip(Transformations, ['sol_T_MH','sol_T_MV','sol_T_MD']):
        Isol_mirror = np.zeros(Isol.shape)
        for i in range(npar):
            print('mirroying ', i)
            usol.vector().set_local(Isol[i,:])
            Isol_mirror[i,:] = interpolate(feut.myfog(usol,T),Vref).vector().get_local()[:] 
        
        
        myhd.addDataset(f,Isol_mirror, label_T)
    
    f.close()

#  ================  Extracting Alphas ============================================
if(op == 2):
    os.system('rm ' + nameYlist.format(labelSnaps))
    # Wbasis = myhd.loadhd5(nameWbasis, 'Wbasis')
    Wbasis_M = myhd.loadhd5(nameWbasis, ['Wbasis','massMatrix'])
    Isol = myhd.loadhd5(nameSnaps.format(labelSnaps),'solutions_trans')
    myhd.savehd5(nameYlist,gdb.getAlphas_fast(Wbasis_M,Isol,npar,Nmax, dotProduct, Vref, dsRef),'Ylist',mode='w') 

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

# ======================= Computing basis =======================================
if(op == 5): 
    os.system('rm ' + nameWbasis)
    Isol = myhd.loadhd5(nameSnaps,'solutions_trans')
    Wbasis_M, f = myhd.zeros_openFile(nameWbasis, [(Nmax,Vref.dim()),[Vref.dim(),Vref.dim()]], ['Wbasis','massMatrix'])
    Wbasis , M = Wbasis_M
    sig, U = gdb.computingBasis_svd(Wbasis, M, Isol,Nmax,Vref, dsRef, dotProduct)
    print(U.shape)
    os.system('rm ' + folder + 'eigens.hd5')
    myhd.savehd5(folder + 'eigens.hd5', [sig,U],['eigenvalues','eigenvectors'], mode = 'w-')
    f.close()

# ======================= Create XY =======================================
if(op == 7): 
    os.system('rm ' + nameXYlist)
    Y = myhd.loadhd5(nameYlist,'Ylist')
    X = myhd.loadhd5(nameEllipseData,'ellipseData')[:,:,2]
    myhd.savehd5(nameXYlist, [Y,X], ['Y','X'], mode='w')