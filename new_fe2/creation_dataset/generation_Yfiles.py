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

f = open("../../../rootDataPath.txt")
rootData = f.read()[:-1]
f.close()

folder = rootData + "/new_fe2/dataset/"

suffix = '_test'
nameSnaps = folder + 'snapshots%s.h5'%suffix
nameMeshRefBnd = folder + 'boundaryMesh.xdmf'
nameWbasis = folder + 'Wbasis.h5'
nameYlist = folder + 'Y%s.h5'%suffix
nameXYlist = folder + 'XY%s.hd5'%suffix
nameParamRVEdataset = folder + 'paramRVEdataset%s.hd5'%suffix

dotProduct = lambda u,v, dx : assemble(inner(u,v)*ds)

Mref = meut.EnrichedMesh(nameMeshRefBnd)
Vref = VectorFunctionSpace(Mref,"CG", 1)

dxRef = Measure('dx', Mref) 
dsRef = Measure('ds', Mref) 
Nh = Vref.dim()

Nmax = 160

op = int(input('option'))
   
# =============================== Translating solution (Partially) ================================
if(op == 0):
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
            T = feut.affineTransformationExpression(Isol_a[i,:],np.zeros((2,2)), Mref) # B = 0
            Isol_trans[i,:] = Isol_trans[i,:] + interpolate(T,Vref).vector().get_local()[:] 
            
        myhd.addDataset(fIsol,Isol_trans, 'solutions_trans_partial_%s'%load_flag)
        fIsol.close()

# ======================= Computing basis =======================================
if(op == 1): 
    os.system('rm ' + nameWbasis)
    Wbasis_fields, f = myhd.zeros_openFile(nameWbasis, [(Nmax,Nh),(Nmax,Nh),(Nmax,),(Nmax,),(Nh,Nh)], 
                                                       ['Wbasis_A', 'Wbasis_S','sig_A', 'sig_S','massMat'])
    Wbasis_A , Wbasis_S, sig_A, sig_S, Mmat = Wbasis_fields
    for load_flag, Wbasis, sig in zip(['A', 'S'],[Wbasis_A , Wbasis_S], [sig_A,sig_S]):
        Isol = myhd.loadhd5(nameSnaps,'solutions_trans_partial_%s'%load_flag)
        sig[:] = gdb.computingBasis_svd(Wbasis, Mmat, Isol,Nmax,Vref, dsRef, dotProduct)[0][:Nmax] # Mmat equal to both
    
    f.close()

#  ================  Extracting Alphas ============================================
if(op == 2):
    os.system('rm ' + nameYlist)
    for load_flag in ['A', 'S']:
        Wbasis_M = myhd.loadhd5(nameWbasis, ['Wbasis_%s'%load_flag,'massMat'])
        Isol = myhd.loadhd5(nameSnaps,'solutions_trans_partial_%s'%load_flag)
        ns = len(Isol)
        Ylist = gdb.getAlphas_fast(Wbasis_M,Isol,ns,Nmax, dotProduct, Vref, dsRef)
        if os.path.exists(nameYlist):
            fIsol = myhd.loadhd5_openFile(nameYlist, [], mode = 'a')[1]
            myhd.addDataset(fIsol,Ylist,'Ylist_%s'%load_flag)
            fIsol.close()
        else:
            myhd.savehd5(nameYlist,Ylist,'Ylist_%s'%load_flag, mode='w') 
    


# ======================= Create XY =======================================
if(op == 3): 
    os.system('rm ' + nameXYlist)
    Y = myhd.loadhd5(nameYlist,['Ylist_%s'%s for s in ['A','S']])
    X = myhd.loadhd5(nameParamRVEdataset,'param')[:,:,2]
    myhd.savehd5(nameXYlist, Y + [X], ['Y_%s'%s for s in ['A','S']] + ['X'], mode='w')
    os.system('rm ' + nameYlist)
