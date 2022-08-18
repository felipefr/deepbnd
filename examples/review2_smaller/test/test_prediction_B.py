#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 11:45:09 2022

@author: felipefr
"""

import os, sys
import matplotlib.pyplot as plt
import numpy as np
import dolfin as df

from deepBND.__init__ import *
from fetricks.fenics.mesh.mesh import Mesh 
import fetricks.data_manipulation.wrapper_h5py as myhd
import fetricks as ft

# Test Loading 
problemType = ''

folder = rootDataPath + '/review2_smaller/'
folderDataset = folder + 'dataset/'
folderPrediction = folder + 'prediction/'


bndMeshname = folderPrediction + 'boundaryMesh.xdmf'
paramRVEname_test = folderPrediction +  'paramRVEdataset_test.hd5'
snapshotsname = folderDataset +  'snapshots.hd5'
BCname = folderPrediction + 'bcs_fluctuations_big_test.hd5'

Mref = Mesh(bndMeshname)
Vref = df.VectorFunctionSpace(Mref,"CG", 2)

dsRef = df.Measure('ds', Mref) 


ids_test = myhd.loadhd5(paramRVEname_test, 'ids').astype('int')
B_target = myhd.loadhd5(snapshotsname, 'B_A')[ids_test] 
sol_target = myhd.loadhd5(snapshotsname, 'solutions_translation_A')[ids_test]   

u0_p, u1_p, u2_p = myhd.loadhd5(BCname, ['u0', 'u1', 'u2'])

# ns = ids_test.shape[0]
ns = 4 


normal = df.FacetNormal(Mref)
volL = 4.0
u = df.Function(Vref)

for i in range(ns):
    u.vector().set_local(u0_p[i,:])
    B = - ft.Integral(df.outer(u,normal), dsRef, (2,2))/volL
    u.vector().set_local(sol_target[i,:])
    B_target_i = - ft.Integral(df.outer(u,normal), Mref.ds, (2,2))/volL
    
    print("B = ", B)
    print("B_target_i = ", B_target_i)
    print("B_target = ", B_target [i,:,:])
    
    
