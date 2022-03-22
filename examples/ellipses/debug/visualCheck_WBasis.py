#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 15:32:44 2022

@author: felipe
"""


import os, sys
import matplotlib.pyplot as plt
import numpy as np

from deepBND.__init__ import * 

from deepBND.core.fenics_tools.enriched_mesh import EnrichedMesh 
import deepBND.core.data_manipulation.wrapper_h5py as myhd
import deepBND.core.fenics_tools.misc as feut


import dolfin as df # apparently this import should be after the tensorflow stuff

# folder = rootDataPath + "/circles/"
# folderBasis = folder + ''
folder = rootDataPath + "/ellipses/"
folderBasis = folder + 'dataset_cluster/'


nameMeshRefBnd = folderBasis + 'boundaryMesh.xdmf'
nameWbasis = folderBasis +  'Wbasis.hd5'

W_A = myhd.loadhd5(nameWbasis, 'Wbasis_A')
W_S = myhd.loadhd5(nameWbasis, 'Wbasis_S')

Mref = EnrichedMesh(nameMeshRefBnd)
Vref = df.VectorFunctionSpace(Mref,"CG", 1)
v2d = df.vertex_to_dof_map(Vref)
normal = df.FacetNormal(Mref)


Nb = 4

W_A_list = []
W_S_list = []

for i in range(Nb):
    W_A_list.append(df.Function(Vref))
    W_S_list.append(df.Function(Vref))
    W_A_list[-1].rename("W_A_%d"%i, "blabla")
    W_S_list[-1].rename("W_S_%d"%i, "blabla")
    
    w_temp = W_A[i,:]     
    w_temp[v2d[160:]] = 0.0

    W_A_list[-1].vector().set_local(w_temp)

    w_temp = W_S[i,:]     
    w_temp[v2d[160:]] = 0.0

    W_S_list[-1].vector().set_local(w_temp)

    I_S = feut.Integral(df.outer(W_S_list[-1], normal), Mref.ds, (2,2))
    I_A = feut.Integral(df.outer(W_A_list[-1], normal), Mref.ds, (2,2))
    
    print(I_S, I_A)

    I_S = feut.Integral(W_S_list[-1], Mref.dx, (2,))
    I_A = feut.Integral(W_A_list[-1], Mref.dx, (2,))
    
    print(I_S, I_A)

with df.XDMFFile("Wbasis_circles.xdmf") as f:
    f.parameters["flush_output"] = True
    f.parameters["functions_share_mesh"] = True
    
    for w in W_A_list:    
        f.write(w, 0.)

    for w in W_S_list:    
        f.write(w, 0.)
        