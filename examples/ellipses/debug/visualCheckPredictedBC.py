#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 18:40:49 2022

@author: felipe
"""

import os, sys
import matplotlib.pyplot as plt
import numpy as np

from deepBND.__init__ import * 

from deepBND.core.fenics_tools.enriched_mesh import EnrichedMesh 
import deepBND.core.data_manipulation.wrapper_h5py as myhd


import dolfin as df # apparently this import should be after the tensorflow stuff

archId = 'big_classical'
Nrb = 140
nX = 72

suffixParam = '_test'
suffix = '_test'


folder = rootDataPath + "/ellipses/"
folderTrain = folder + 'training_cluster/'
folderBasis = folder + 'dataset_cluster/'
folderPrediction = folder + "prediction_cluster/"
nameMeshRefBnd = folderBasis + 'boundaryMesh.xdmf'
nameWbasis = folderBasis +  'Wbasis.hd5'
paramRVEname = folderPrediction + 'paramRVEdataset{0}.hd5'.format(suffixParam)
bcs_namefile = folderPrediction + 'bcs_{0}_{1}{2}.hd5'.format(archId, Nrb, suffix)
nameScaleXY = {}


u0_p = myhd.loadhd5(bcs_namefile, 'u0')
u1_p = myhd.loadhd5(bcs_namefile, 'u1')
u2_p = myhd.loadhd5(bcs_namefile, 'u2')

Mref = EnrichedMesh(nameMeshRefBnd)
Vref = df.VectorFunctionSpace(Mref,"CG", 1)


uh0 = df.Function(Vref)
uh1 = df.Function(Vref)
uh2 = df.Function(Vref)

# d2v = df.dof_to_vertex_map(Vref)
v2d = df.vertex_to_dof_map(Vref)

i = 2

w0 = u0_p[i,:]
w1 = u1_p[i,:]
w2 = u2_p[i,:]

w0[v2d[160:]] = 0.0
w1[v2d[160:]] = 0.0
w2[v2d[160:]] = 0.0


uh0.vector().set_local(w0)
uh1.vector().set_local(w1)
uh2.vector().set_local(w2)

uh0.rename("u0", "blabla")
uh1.rename("u1", "blabla")
uh2.rename("u2", "blabla")

with df.XDMFFile("boundary.xdmf") as f:
    f.parameters["flush_output"] = True
    f.parameters["functions_share_mesh"] = True

    f.write(uh0, 0.)
    f.write(uh1, 0.)
    f.write(uh2, 0.)