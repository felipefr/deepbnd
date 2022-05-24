"""
Inputs = paramRVEdataset.hd5, snapshots.hd5 (described before) 
Outputs: 
- Wbasis.hd5: Reduced-Basis matrices and singular values. Same comment for labels 'A' and 'S' (above) applies.  
- XY.hd5: Assemble important data of paramRVEdataset.hd5 (to be used as the input of NN training) and also the projected solutions of the snapshots (snapshots.hd5) onto the RB (Wbasis.hd5). Same comment for labels 'A' and 'S' (above) applies.

Obs: Wbasis.hd5 should be obtained to the larger dataset (usually the training one), then should be reused to obtain XY.hd5 files of the remaining datasets. 

The typical order of execution is with op= 0, 1, 2, 3 (one after another). Op = 1 (RB basis obtention) can be skipped according to the situation. 

"""

import sys, os
import numpy as np
from dolfin import *
from timeit import default_timer as timer
import matplotlib.pyplot as plt

from deepBND.__init__ import *
import deepBND.creation_model.RB.RB_utils as rbut
import deepBND.core.multiscale.misc as mtsm

from fetricks.fenics.mesh.mesh import Mesh 
import fetricks.data_manipulation.wrapper_h5py as myhd

dotProduct = lambda u,v, dx_ref: inner(u,v)*dx_ref

def translateSolution(nameSnaps, Vref):
    for load_flag in ['A', 'S']:
        labels = ['solutions_%s'%load_flag,'a_%s'%load_flag,'B_%s'%load_flag]
        Isol, fIsol = myhd.loadhd5_openFile(nameSnaps,labels, mode = 'a')
        Isol_full , Isol_a, Isol_B = Isol # a = -avg(disp), B = - avg(strain) 
        Isol_trans = Isol_full[:,:]
        usol = Function(Vref)
        ns = len(Isol_trans)
        for i in range(ns):
            print('translating ', i)
            usol.vector().set_local(Isol_full[i,:])
            T = mtsm.affineTransformationExpression(Isol_a[i,:], Isol_B[i,:,:], Mref) # B = 0
            Isol_trans[i,:] = Isol_trans[i,:] + interpolate(T,Vref).vector().get_local()[:] 
            
        myhd.addDataset(fIsol,Isol_trans, 'solutions_fluctuations_%s'%load_flag)
        fIsol.close()


def computingBasis(nameSnaps, nameWbasis, Nmax, Nh, Vref, dsRef):
    os.system('rm ' + nameWbasis)
    Wbasis_fields, f = myhd.zeros_openFile(nameWbasis, [(Nmax,Nh),(Nmax,Nh),(Nmax,),(Nmax,),(Nh,Nh)], 
                                                       ['Wbasis_A', 'Wbasis_S','sig_A', 'sig_S','massMat'])
    Wbasis_A , Wbasis_S, sig_A, sig_S, Mmat = Wbasis_fields
    for load_flag, Wbasis, sig in zip(['A', 'S'],[Wbasis_A , Wbasis_S], [sig_A,sig_S]):
        Isol = myhd.loadhd5(nameSnaps,'solutions_fluctuations_%s'%load_flag)
        sig[:] = rbut.computingBasis_svd(Wbasis, Mmat, Isol,Nmax,Vref, dsRef, dotProduct)[0][:Nmax] # Mmat equal to both
    
    f.close()


def extractAlpha(nameSnaps, nameWbasis, Nmax, nameYlist, Vref, dsRef):
    os.system('rm ' + nameYlist)
    for load_flag in ['A', 'S']:
        Wbasis_M = myhd.loadhd5(nameWbasis, ['Wbasis_%s'%load_flag,'massMat'])
        Isol = myhd.loadhd5(nameSnaps,'solutions_fluctuations_%s'%load_flag)
        ns = len(Isol)
        Ylist = rbut.getAlphas_fast(Wbasis_M,Isol, ns, Nmax, dotProduct, Vref, dsRef)
        if os.path.exists(nameYlist):
            fIsol = myhd.loadhd5_openFile(nameYlist, [], mode = 'a')[1]
            myhd.addDataset(fIsol,Ylist,'Ylist_%s'%load_flag)
            fIsol.close()
        else:
            myhd.savehd5(nameYlist,Ylist,'Ylist_%s'%load_flag, mode='w') 
    


def createXY(nameParamRVEdataset, nameYlist, nameXYlist, id_feature = 2):
    
    os.system('rm ' + nameXYlist)
    Y = myhd.loadhd5(nameYlist,['Ylist_%s'%s for s in ['A','S']])
    X = myhd.loadhd5(nameParamRVEdataset,'param')[:,:, id_feature]
    X = X.reshape((X.shape[0],-1))
    myhd.savehd5(nameXYlist, Y + [X], ['Y_%s'%s for s in ['A','S']] + ['X'], mode='w')
    os.system('rm ' + nameYlist)
    


if __name__ == '__main__': 
    
    folder = rootDataPath + "/review2/dataset/"
    folder_mesh = rootDataPath + "/review2/dataset/"
    
    suffix = ''
    nameSnaps = folder + 'snapshots%s.hd5'%suffix
    nameMeshRefBnd = folder_mesh + 'boundaryMesh.xdmf'
    nameWbasis = folder + 'Wbasis.hd5'
    nameYlist = folder + 'Y%s.hd5'%suffix
    nameXYlist = folder + 'XY%s.hd5'%suffix
    nameParamRVEdataset = folder + 'paramRVEdataset%s.hd5'%suffix
    
    Mref = Mesh(nameMeshRefBnd)
    Vref = VectorFunctionSpace(Mref,"CG", 2)
    
    dxRef = Measure('dx', Mref) 
    dsRef = Measure('ds', Mref) 
    Nh = Vref.dim()
    
    Nmax = 10
    
    op = int(input('option (0 to 3)'))
        
    if(op==0):
        translateSolution(nameSnaps, Vref)
    elif(op==1):
        computingBasis(nameSnaps, nameWbasis, Nmax, Nh, Vref, dsRef)
    elif(op==2):
        extractAlpha(nameSnaps, nameWbasis, Nmax, nameYlist, Vref, dsRef)
    elif(op==3):
        createXY(nameParamRVEdataset, nameYlist, nameXYlist, id_feature = [0,1])    
    elif(op==4): # train,  validation, test splitting
        ns = len(ids)
        seed = 2
        np.random.seed(seed)
        shuffled_ids = np.arange(0,ns) 
        np.random.shuffle(shuffled_ids) 
        
        id_val = np.arange(0, int(np.floor(r_val*ns))).astype('int')
        id_test = np.arange(id_val[-1] + 1, int(np.floor((r_val+r_test)*ns)))
        id_train = np.arange(id_test[-1], ns)
        
        id_val = shuffled_ids[id_val]
        id_test = shuffled_ids[id_test]
        id_train = shuffled_ids[id_train]
        
        new_snaps_fields_val = []
        new_snaps_fields_test = []
        new_snaps_fields_train = []
        
        for f in snaps:
            new_snaps_fields_val.append(f[id_val])
            new_snaps_fields_test.append(f[id_test])
            new_snaps_fields_train.append(f[id_train])
        
        myhd.savehd5("paramRVEdataset_val.hd5", [ids[id_val], param[id_val]], ["id", "param"] , "w-")
        myhd.savehd5("paramRVEdataset_test.hd5", [ids[id_test], param[id_test]], ["id", "param"] , "w-")
        myhd.savehd5("paramRVEdataset_train.hd5", [ids[id_train], param[id_train]], ["id", "param"] , "w-")
        
        myhd.savehd5("snapshots_val.hd5", new_snaps_fields_val, labels, "w-")
                