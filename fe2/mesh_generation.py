#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Availabel in: https://github.com/felipefr/micmacsFenics.git
@author: Felipe Figueredo Rocha, f.rocha.felipe@gmail.com, felipe.figueredorocha@epfl.ch
   
"""

import sys, os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import dolfin as df 
import matplotlib.pyplot as plt
from ufl import nabla_div
sys.path.insert(0, '/home/felipefr/github/micmacsFenics/utils/')
sys.path.insert(0,'../utils/')

import multiscaleModels as mscm
from fenicsUtils import symgrad, symgrad_voigt, Integral
import numpy as np


import fenicsMultiscale as fmts
import myHDF5 as myhd
import meshUtils as meut
import elasticity_utils as elut
from tensorflow_for_training import *
import tensorflow as tf
import symmetryLib as symlpy
from timeit import default_timer as timer
import multiphenics as mp

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_ranks = comm.Get_size()

print('rank, num_ranks ', rank, num_ranks)

permY = np.array([2,0,3,1,12,10,8,4,13,5,14,6,15,11,9,7,30,28,26,24,22,16,31,17,32,18,33,19,34,20,35,29,27,25,23,21])

def write_mesh(ellipseData, nameMesh, size = 'reduced'):
    maxOffset = 2
    
    H = 1.0 # size of each square
    NxL = NyL = 2
    NL = NxL*NyL
    x0L = y0L = -H 
    LxL = LyL = 2*H
    # lcar = (2/30)*H
    lcar = (2/30)*H
    Nx = (NxL+2*maxOffset)
    Ny = (NyL+2*maxOffset)
    Lxt = Nx*H
    Lyt = Ny*H
    NpLxt = int(Lxt/lcar) + 1
    NpLxL = int(LxL/lcar) + 1
    print("NpLxL=", NpLxL) 
    x0 = -Lxt/2.0
    y0 = -Lyt/2.0
    r0 = 0.2*H
    r1 = 0.4*H
    Vfrac = 0.282743
    rm = H*np.sqrt(Vfrac/np.pi)

    if(size == 'reduced'):
        meshGMSH = meut.ellipseMesh2(ellipseData[:4,:], x0L, y0L, LxL, LyL, lcar)
        meshGMSH.setTransfiniteBoundary(NpLxL)
        
    elif(size == 'full'):
        meshGMSH = meut.ellipseMesh2Domains(x0L, y0L, LxL, LyL, NL, ellipseData[:36,:], Lxt, Lyt, lcar, x0 = x0, y0 = y0)
        meshGMSH.setTransfiniteBoundary(NpLxt)
        meshGMSH.setTransfiniteInternalBoundary(NpLxL)   
            
    meshGMSH.write(nameMesh, opt = 'fenics')
    
# loading boundary reference mesh
# nameMeshRefBnd = 'boundaryMesh.xdmf'
# Mref = meut.EnrichedMesh(nameMeshRefBnd)
# Vref = df.VectorFunctionSpace(Mref,"CG", 1)

# dxRef = df.Measure('dx', Mref) 
    
# loading the DNN model
folder = './models/dataset_axial3/'
ellipseData = myhd.loadhd5(folder +  'ellipseData.h5', 'ellipseData')

# piola_mat = syml.PiolaTransform_matricial('mHalfPi', Vref)

# defining the micro model
i = 0
nCells = 80

# meshMicro = get_mesh(ellipseData[i], './meshes/mesh_micro.xml', 'reduced')
for i in range(20,nCells):
    if(i%num_ranks == rank):
        write_mesh(ellipseData[i], './meshes/mesh_micro_{0}_full.xdmf'.format(i), 'full')