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
import deepBND.core.data_manipulation.wrapper_h5py as myhd
from deepBND.core.fenics_tools.enriched_mesh import EnrichedMesh 
import deepBND.core.multiscale.misc as mtsm

dotProduct = lambda u,v, dx : assemble(inner(u,v)*ds)

def translateSolution(nameSnaps, Vref):
    for load_flag in ['A', 'S']:
        labels = ['solutions_%s'%load_flag,'a_%s'%load_flag,'B_%s'%load_flag]
        Isol, fIsol = myhd.loadhd5_openFile(nameSnaps,labels, mode = 'a')
        Isol_full , Isol_a, Isol_B = Isol
        Isol_trans = Isol_full[:,:]
        usol = Function(Vref)
        ns = len(Isol_trans)
        for i in range(ns):
            print('translating ', i)
            usol.vector().set_local(Isol_full[i,:])
            T = mtsm.affineTransformationExpression(Isol_a[i,:],np.zeros((2,2)), Mref) # B = 0
            Isol_trans[i,:] = Isol_trans[i,:] + interpolate(T,Vref).vector().get_local()[:] 
            
        myhd.addDataset(fIsol,Isol_trans, 'solutions_trans_partial_%s'%load_flag)
        fIsol.close()


def computingBasis(nameSnaps, nameWbasis, Nmax, Nh, Vref, dsRef):
    os.system('rm ' + nameWbasis)
    Wbasis_fields, f = myhd.zeros_openFile(nameWbasis, [(Nmax,Nh),(Nmax,Nh),(Nmax,),(Nmax,),(Nh,Nh)], 
                                                       ['Wbasis_A', 'Wbasis_S','sig_A', 'sig_S','massMat'])
    Wbasis_A , Wbasis_S, sig_A, sig_S, Mmat = Wbasis_fields
    for load_flag, Wbasis, sig in zip(['A', 'S'],[Wbasis_A , Wbasis_S], [sig_A,sig_S]):
        Isol = myhd.loadhd5(nameSnaps,'solutions_trans_partial_%s'%load_flag)
        sig[:] = rbut.computingBasis_svd(Wbasis, Mmat, Isol,Nmax,Vref, dsRef, dotProduct)[0][:Nmax] # Mmat equal to both
    
    f.close()


def extractAlpha(nameSnaps, nameWbasis, Nmax, nameYlist, Vref, dsRef):
    os.system('rm ' + nameYlist)
    for load_flag in ['A', 'S']:
        Wbasis_M = myhd.loadhd5(nameWbasis, ['Wbasis_%s'%load_flag,'massMat'])
        Isol = myhd.loadhd5(nameSnaps,'solutions_trans_partial_%s'%load_flag)
        ns = len(Isol)
        Ylist = rbut.getAlphas_fast(Wbasis_M,Isol, ns, Nmax, dotProduct, Vref, dsRef)
        if os.path.exists(nameYlist):
            fIsol = myhd.loadhd5_openFile(nameYlist, [], mode = 'a')[1]
            myhd.addDataset(fIsol,Ylist,'Ylist_%s'%load_flag)
            fIsol.close()
        else:
            myhd.savehd5(nameYlist,Ylist,'Ylist_%s'%load_flag, mode='w') 
    


def createXY(nameParamRVEdataset, nameYlist, nameXYlist):
    
    os.system('rm ' + nameXYlist)
    Y = myhd.loadhd5(nameYlist,['Ylist_%s'%s for s in ['A','S']])
    X = myhd.loadhd5(nameParamRVEdataset,'param')[:,:,2]
    myhd.savehd5(nameXYlist, Y + [X], ['Y_%s'%s for s in ['A','S']] + ['X'], mode='w')
    os.system('rm ' + nameYlist)
    


if __name__ == '__main__': 
    
    folder = rootDataPath + "/deepBND/dataset/"
    
    suffix = '_validation'
    nameSnaps = folder + 'snapshots%s.hd5'%suffix
    nameMeshRefBnd = folder + 'boundaryMesh.xdmf'
    nameWbasis = folder + 'Wbasis.hd5'
    nameYlist = folder + 'Y%s.h5'%suffix
    nameXYlist = folder + 'XY%s.hd5'%suffix
    nameParamRVEdataset = folder + 'paramRVEdataset%s.hd5'%suffix
    
    Mref = EnrichedMesh(nameMeshRefBnd)
    Vref = VectorFunctionSpace(Mref,"CG", 1)
    
    dxRef = Measure('dx', Mref) 
    dsRef = Measure('ds', Mref) 
    Nh = Vref.dim()
    
    Nmax = 160
    
    op = int(input('option (0 to 3)'))
    
    if(op==0):
        translateSolution(nameSnaps, Vref)
    elif(op==1):
        computingBasis(nameSnaps, nameWbasis, Nmax, Nh, Vref, dsRef)
    elif(op==2):
        extractAlpha(nameSnaps, nameWbasis, Nmax, nameYlist, Vref, dsRef)
    elif(op==3):
        createXY(nameParamRVEdataset, nameYlist, nameXYlist)
        