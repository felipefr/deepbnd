"""
This file is part of deepBND, a data-driven enhanced boundary condition implementaion for 
computational homogenization problems, using RB-ROM and Neural Networks.
Copyright (c) 2020-2023, Felipe Rocha.
See file LICENSE.txt for license information. 
Please cite this work according to README.md.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or <felipe.f.rocha@gmail.com>
"""

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
import dolfin as df

os.environ['HDF5_DISABLE_VERSION_CHECK']='2'

from deepBND.__init__ import *
import deepBND.creation_model.RB.RB_utils as rbut
import deepBND.core.multiscale.misc as mtsm

from fetricks.fenics.mesh.mesh import Mesh 
import fetricks.data_manipulation.wrapper_h5py as myhd
import deepBND.core.data_manipulation.utils as dman

dotProduct = lambda u,v, dx_ref: df.inner(u,v)*dx_ref

def translateSolution(nameSnaps, Vref):
    for load_flag in ['A', 'S']:
        labels = ['solutions_%s'%load_flag,'a_%s'%load_flag,'B_%s'%load_flag]
        Isol, fIsol = myhd.loadhd5_openFile(nameSnaps,labels, mode = 'a')
        Isol_full , Isol_a, Isol_B = Isol # a = -avg(disp), B = - avg(strain) 
        Isol_trans = Isol_full[:,:]
        Isol_fluc = Isol_full[:,:]
        usol = df.Function(Vref)
        ns = len(Isol_trans)
        for i in range(ns):
            print('translating ', i)
            usol.vector().set_local(Isol_full[i,:])
            Ttrans = mtsm.affineTransformationExpression(Isol_a[i,:], np.zeros((2,2)), Mref) # B = 0
            Tfluc = mtsm.affineTransformationExpression(Isol_a[i,:], Isol_B[i,:,:], Mref) # B = 0
            Isol_fluc[i,:] = Isol_fluc[i,:] + df.interpolate(Tfluc,Vref).vector().get_local()[:] 
            Isol_trans[i,:] = Isol_trans[i,:] + df.interpolate(Ttrans,Vref).vector().get_local()[:] 
    
        myhd.addDataset(fIsol,Isol_fluc, 'solutions_fluctuations_%s'%load_flag)
        myhd.addDataset(fIsol,Isol_trans, 'solutions_translation_%s'%load_flag)
                        
        fIsol.close()


def computingBasis(nameSnaps, nameWbasis, Nmax, Nh, Vref, dsRef, solution_label):
    os.system('rm ' + nameWbasis)
    Wbasis_fields, f = myhd.zeros_openFile(nameWbasis, [(Nmax,Nh),(Nmax,Nh),(Nmax,),(Nmax,),(Nh,Nh)], 
                                                       ['Wbasis_A', 'Wbasis_S','sig_A', 'sig_S','massMat'])
    Wbasis_A , Wbasis_S, sig_A, sig_S, Mmat = Wbasis_fields
    for load_flag, Wbasis, sig in zip(['A', 'S'],[Wbasis_A , Wbasis_S], [sig_A,sig_S]):
        Isol = myhd.loadhd5(nameSnaps,'%s_%s'%(solution_label, load_flag) )
        sig[:] = rbut.computingBasis_svd(Wbasis, Mmat, Isol,Nmax,Vref, dsRef, dotProduct)[0][:Nmax] # Mmat equal to both
    
    f.close()


def extractAlpha(nameSnaps, nameWbasis, Nmax, nameYlist, Vref, dsRef, solution_label):
    os.system('rm ' + nameYlist)
    for load_flag in ['A', 'S']:
        Wbasis_M = myhd.loadhd5(nameWbasis, ['Wbasis_%s'%load_flag,'massMat'])
        Isol = myhd.loadhd5(nameSnaps, '%s_%s'%(solution_label, load_flag))
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
    ids = myhd.loadhd5(nameParamRVEdataset,'ids')[:]
    X = X.reshape((X.shape[0],-1))
    myhd.savehd5(nameXYlist, Y + [X] + [ids], ['Y_%s'%s for s in ['A','S']] + ['X', 'ids'], mode='w')
    os.system('rm ' + nameYlist)
    


if __name__ == '__main__': 
    
    folder = rootDataPath + "/review2_smaller/dataset/"
    folder_mesh = rootDataPath + "/review2_smaller/dataset/"
    
    suffix = '_translation'
    nameSnaps = folder + 'snapshots.hd5'
    nameMeshRefBnd = folder_mesh + 'boundaryMesh.xdmf'
    nameWbasis = folder + 'Wbasis%s.hd5'%suffix
    nameYlist = folder + 'Y%s.hd5'%suffix
    nameXYlist = folder + 'XY%s.hd5'%suffix
    nameScaler = folder + 'scaler%s_{0}.hd5'%suffix
    nameParamRVEdataset = folder + 'paramRVEdataset.hd5'
    
    Mref = Mesh(nameMeshRefBnd)
    Vref = df.VectorFunctionSpace(Mref,"CG", 2)
    
    dxRef = df.Measure('dx', Mref) 
    dsRef = df.Measure('ds', Mref) 
    Nh = Vref.dim()
    print(Nh)
    
    Nmax = 800
    
    op = int(input('option (0 to 3)'))
    # op = 4
        
    if(op==0):
        translateSolution(nameSnaps, Vref)
    elif(op==1):
        computingBasis(nameSnaps, nameWbasis, Nmax, Nh, Vref, dsRef, "solutions%s"%suffix)
    elif(op==2):
        extractAlpha(nameSnaps, nameWbasis, Nmax, nameYlist, Vref, dsRef, "solutions%s"%suffix)
    elif(op==3):
        createXY(nameParamRVEdataset, nameYlist, nameXYlist, id_feature = [0,1])    
    elif(op==4): # train,  validation, test splitting
            
    
        ns = len(myhd.loadhd5(nameXYlist, "ids"))
        seed = 2
        np.random.seed(seed)
        shuffled_ids = np.arange(0,ns) 
        np.random.shuffle(shuffled_ids)
        
        r_val = 0.05
        r_test = 0.025
        
        id_val = np.arange(0, int(np.floor(r_val*ns))).astype('int')
        id_test = np.arange(id_val[-1] + 1, int(np.floor((r_val+r_test)*ns)))
        id_train = np.arange(id_test[-1], ns)
        
        id_val = shuffled_ids[id_val]
        id_test = shuffled_ids[id_test]
        id_train = shuffled_ids[id_train]
        
        labels = ['X', 'Y_A', 'Y_S']
        X, Y_A, Y_S = myhd.loadhd5(nameXYlist, labels )  
        
        myhd.savehd5(nameXYlist.split('.')[0] + "_val.hd5" , [X[id_val], Y_A[id_val], Y_S[id_val]], labels , "w-")
        myhd.savehd5(nameXYlist.split('.')[0] + "_test.hd5", [X[id_test], Y_A[id_test], Y_S[id_test]], labels , "w-")
        myhd.savehd5(nameXYlist.split('.')[0] + "_train.hd5", [X[id_train], Y_A[id_train], Y_S[id_train]], labels , "w-")
        
        
        dman.exportScale(nameXYlist, nameScaler.format('A'), 72, Nmax, Ylabel = 'Y_A', scalerType = 'MinMax11' )
        dman.exportScale(nameXYlist, nameScaler.format('S'), 72, Nmax, Ylabel = 'Y_S', scalerType = 'MinMax11' )
        