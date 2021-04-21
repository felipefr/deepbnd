import sys, os
import numpy as np
sys.path.insert(0, '../../utils/')

import fenicsWrapperElasticity as fela
import generation_deepBoundary_lib as gdb
from dolfin import *
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import myHDF5 as myhd
import meshUtils as meut
import fenicsUtils as feut
import symmetryLib as syml

folder = './models/dataset_partially_axial2/'
folderBasis = './models/dataset_partially_axial1/'


loadType = 'axial'
nameSnaps = folder + 'snapshots.h5'
nameSnaps_original = folder + 'snapshots.h5'
nameMeshRefBnd = 'boundaryMesh.xdmf'
nameWbasis = folderBasis + 'Wbasis.h5'
nameYlist = folder + 'Y.h5'
nameXYlist = folder + 'XY.h5'
nameXYlist_stress = folder + 'XY_stress.h5'
nameTau = folder + 'tau.h5'
nameEllipseData = folder + 'ellipseData.h5'
nameEllipseData_original = folder + 'ellipseData.h5'
nameEigen = folder + 'eigen.hd5'

dotProduct = lambda u,v, dx : assemble(inner(u,v)*ds)

Mref = meut.EnrichedMesh(nameMeshRefBnd)
Vref = VectorFunctionSpace(Mref,"CG", 1)

dxRef = Measure('dx', Mref) 
dsRef = Measure('ds', Mref) 

ns = 5120
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
   

# ====================== Merging Datasets ===========================================
if(op == 0):
    print('merging')
    os.system('rm ' + nameSnaps.format(labelSnaps))
    myhd.merge([nameSnaps.format(i) for i in range(Npartitions)], nameSnaps.format(labelSnaps), 
                InputLabels = ['solutions', 'a','B', 'sigma', 'sigmaTotal'], 
                OutputLabels = ['solutions', 'a','B', 'sigma', 'sigmaTotal'], axis = 0, mode = 'w-')


# =============================== Translating solution ================================
if(op == 1):
    Isol, fIsol = myhd.loadhd5_openFile(nameSnaps.format(labelSnaps),['solutions','a','B'], mode = 'a')
    Isol_full , Isol_a, Isol_B = Isol
    Isol_trans = Isol_full[:,:]
    usol = Function(Vref)
    for i in range(npar):
        print('translating ', i)
        usol.vector().set_local(Isol_full[i,:])
        T = feut.affineTransformationExpression(Isol_a[i,:],Isol_B[i,:,:], Mref)
        Isol_trans[i,:] = Isol_trans[i,:] + interpolate(T,Vref).vector().get_local()[:] 
        
    myhd.addDataset(fIsol,Isol_trans, 'solutions_trans')
    fIsol.close()

# =============================== Translating solution (Partially) ================================
if(op == 10):
    Isol, fIsol = myhd.loadhd5_openFile(nameSnaps.format(labelSnaps),['solutions','a','B'], mode = 'a')
    Isol_full , Isol_a, Isol_B = Isol
    Isol_trans = Isol_full[:,:]
    usol = Function(Vref)
    for i in range(npar):
        print('translating ', i)
        usol.vector().set_local(Isol_full[i,:])
        T = feut.affineTransformationExpression(Isol_a[i,:],np.zeros((2,2)), Mref) # B = 0
        Isol_trans[i,:] = Isol_trans[i,:] + interpolate(T,Vref).vector().get_local()[:] 
        
    myhd.addDataset(fIsol,Isol_trans, 'solutions_trans_partially')
    fIsol.close()



#  ================  Extracting Alphas ============================================
if(op == 2):
    os.system('rm ' + nameYlist.format(labelSnaps))
    # Wbasis = myhd.loadhd5(nameWbasis, 'Wbasis')
    Wbasis_M = myhd.loadhd5(nameWbasis, ['Wbasis','massMatrix'])
    Isol = myhd.loadhd5(nameSnaps.format(labelSnaps),'solutions_trans_partially')
    myhd.savehd5(nameYlist,gdb.getAlphas_fast(Wbasis_M,Isol,npar,Nmax, dotProduct, Vref, dsRef),'Ylist',mode='w') 
    
    if(myhd.checkExistenceDataset(nameSnaps.format(labelSnaps), 'load_sign')):
        load_sign = myhd.loadhd5(nameSnaps.format(labelSnaps), 'load_sign')
        myhd.addDataset(nameYlist, load_sign, 'load_sign')

# ======================= Computing basis =======================================
if(op == 3): 
    os.system('rm ' + nameWbasis)
    Isol = myhd.loadhd5(nameSnaps,'solutions_trans_partially')
    Wbasis_M, f = myhd.zeros_openFile(nameWbasis, [(Nmax,Vref.dim()),[Vref.dim(),Vref.dim()]], ['Wbasis','massMatrix'])
    Wbasis , M = Wbasis_M
    sig, U = gdb.computingBasis_svd(Wbasis, M, Isol,Nmax,Vref, dsRef, dotProduct)
    print(U.shape)
    os.system('rm ' + nameEigen)
    # myhd.savehd5(nameEigen, [sig,U],['eigenvalues','eigenvectors'], mode = 'w-')
    myhd.savehd5(nameEigen, sig,'eigenvalues', mode = 'w-')
    f.close()


# ======================= Computing basis for stress ==============================
if(op == 4):
    E1 = 10.0
    nu = 0.3
    contrast = 0.1 #inverse than in generation
    Nmax = 156
    os.system('rm ' + nameTau)
    Isol = myhd.loadhd5(nameSnaps.format(labelSnaps),'solutions_trans_partially')
    EllipseData = myhd.loadhd5(filename = nameEllipseData, label = 'ellipseData')
    Wbasis = myhd.loadhd5(nameWbasis, 'Wbasis')
    tau, f = myhd.zeros_openFile(nameTau, [(ns,Nmax,3),(ns,3)]  , ['tau', 'tau_0'])
    # gdb.getStressBasis_noMesh(tau,Wbasis, Isol, EllipseData[:ns,:,:], Nmax, Vref, [E1,nu, contrast], EpsDirection = 1) # shear
    gdb.getStressBasis_noMesh_partitioned(tau,Wbasis, Isol, EllipseData[:ns,:,:], Nmax, Vref, [E1,nu, contrast],  1, n0, n1) # shear
    f.close()

# ======================= Create XY =======================================
if(op == 5): 
    os.system('rm ' + nameXYlist)
    Y = myhd.loadhd5(nameYlist,'Ylist')
    # X = myhd.loadhd5(nameEllipseData,'ellipseData')[:,:,2]
    X = myhd.loadhd5(folder + 'XY_temp.h5','X')

    myhd.savehd5(nameXYlist, [Y,X], ['Y','X'], mode='w')
    
    if(myhd.checkExistenceDataset(nameYlist, 'load_sign')):
        load_sign = myhd.loadhd5(nameYlist, 'load_sign')
        myhd.addDataset(nameXYlist, load_sign, 'load_sign')

# ============================ Transforming solution =============================================    
if(op == 6):
    IsolOriginal, fOriginal = myhd.loadhd5_openFile(nameSnaps_original.format(labelSnaps),'solutions_trans_partially', mode = 'r')
    ns, ndim = IsolOriginal.shape
    
    if(loadType == 'shear'):
        Tlabels = ['id', 'horiz', 'vert', 'diag', 'halfPi' , 'mHalfPi']
    elif(loadType == 'axial'): 
        Tlabels = ['id', 'horiz', 'vert', 'diag']
    elif(loadType == 'shear_limited'):
        Tlabels = ['id']

        
    Ntransformation = len(Tlabels)
    
    Isol = np.zeros((Ntransformation*ns,ndim))
    loadSign =np.zeros(Ntransformation*ns)
    
    for i, label in enumerate(Tlabels):
        Piola_mat = syml.PiolaTransform_matricial(label , Vref)
        loadSign_label = syml.getLoadSign(label,loadType[:5])
        for j in range(ns):
            print('trasforming {0} as T_{1} sign {2}'.format(j,i,loadSign_label) )
            Isol[i*ns+j,:] = loadSign_label*Piola_mat@IsolOriginal[j,:]
            loadSign[i*ns+j] = loadSign_label
            

    os.system('rm ' + nameSnaps.format(labelSnaps))    
    myhd.savehd5(nameSnaps.format(labelSnaps) , [Isol, loadSign],  ['solutions_trans_partially','load_sign'], mode = 'w')

# ============================ Transforming Input =============================================    
if(op == 8):
    ellipseDataOriginal = myhd.loadhd5(nameEllipseData_original.format(labelSnaps),'ellipseData')
    ns, Ncircles, Nparam = ellipseDataOriginal.shape
    
    if(loadType == 'shear'):
        Tlabels = ['id', 'horiz', 'vert', 'diag', 'halfPi' , 'mHalfPi']
    elif(loadType == 'axial'): 
        Tlabels = ['id', 'horiz', 'vert', 'diag']
    elif(loadType == 'shear_limited'):
        Tlabels = ['id']

        
    Ntransformation = len(Tlabels)
    
    ellipseData = np.zeros((Ntransformation*ns,Ncircles, Nparam))
    loadSign =np.zeros(Ntransformation*ns)
    
    for i, label in enumerate(Tlabels):
        perm = syml.getPermutation(label)
        ellipseData[i*ns:(i+1)*ns,:,:] = ellipseDataOriginal[:,:,:] 
        ellipseData[i*ns:(i+1)*ns,:,2] = ellipseDataOriginal[:,perm,2] 
        loadSign[i*ns:(i+1)*ns] = syml.getLoadSign(label,loadType[:5])

    os.system('rm ' + nameEllipseData.format(labelSnaps))    
    myhd.savehd5(nameEllipseData.format(labelSnaps) , [ellipseData, loadSign],  ['ellipseData','load_sign'], mode = 'w')

    
# ======================== Merging Y ==================================================
if(op == 7):
    print('merging Y')
    os.system('rm ' + nameYlist.format(labelSnaps))
    myhd.merge([nameYlist.format(i) for i in range(Npartitions)], nameYlist.format(labelSnaps), 
                InputLabels = ['Ylist'], OutputLabels = ['Ylist'], axis = 0, mode = 'w-')


# ======================== Transforming Stress Output ==================================================
if(op == 9):
    print('merging Y')
    Y = myhd.loadhd5(nameSnaps,'sigma')
    X = myhd.loadhd5(nameEllipseData,'ellipseData')[:,:,2]
    # X = myhd.loadhd5(nameXYlist,'X')


    os.system('rm ' + nameXYlist_stress)  
    myhd.savehd5(nameXYlist_stress, [Y,X], ['Y','X'], mode='w')