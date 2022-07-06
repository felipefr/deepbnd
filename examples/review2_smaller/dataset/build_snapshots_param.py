#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 03:41:50 2022

@author: felipe
"""

# Build paramRVEdataset.hd5: It contains geometrical description for each snapshot. It allows the mesh generation, etc. 

import sys, os
import numpy as np

from deepBND.__init__ import *
import deepBND.creation_model.dataset.generation_inclusions as geni
from deepBND.core.multiscale.mesh_RVE import paramRVE_default 
import fetricks.data_manipulation.wrapper_h5py as myhd

def build_paramRVE(paramRVEname, ns, seed):
    gap = 0.0499 # in percetange of H
    Vfrac = 0.12567 # to compute rm
     
    p = paramRVE_default(NxL = 2, NyL = 2, maxOffset = 2, Vfrac = Vfrac)
    NR = p.Nx*p.Ny # 64
    assert(NR == 36)
    
    
    dxy_max = p.H*(0.5 - gap) - p.rm
    
    print(p.H, dxy_max, p.rm)
    
    # Radius Generation
    np.random.seed(seed)
    
    os.system('rm ' + paramRVEname)
    X_ids, f = myhd.zeros_openFile(filename = paramRVEname,  shape = [(ns,NR,5), (ns,)], label = ['param', 'ids'], mode = 'w')
    
    X, ids = X_ids
    
    ellipseData_pattern = geni.getEllipse_emptyRadius(p.Nx,p.Ny,p.Lxt, p.Lyt, p.x0, p.y0)
    
    thetas_X = geni.getScikitoptSample(NR,ns, -1.0, 1.0,  seed, op = 'lhs')
    thetas_Y = geni.getScikitoptSample(NR,ns, -1.0, 1.0,  seed + 1, op = 'lhs')
    
    for i in range(ns):
        print("inserting on ", i)
        ids[i] = i
        X[i,:,:] = ellipseData_pattern
        X[i,:,0] = X[i,:,0] + dxy_max*thetas_X[i,:]
        X[i,:,1] = X[i,:,1] + dxy_max*thetas_Y[i,:]
        X[i,:,2] = p.rm # to give the same area
            
    f.close()


if __name__ == '__main__':
    
    # big dataset
    folder = rootDataPath + "/review2_smaller/dataset/"
    ns = 12
    seed = 2    
    
    # folder = rootDataPath + "/review2_smaller/prediction/"
    # ns = 20
    # seed = 8    
    
    # First generation of 60K simulations was done with seed = 2
    suffix = ""
    paramRVEname = folder + 'paramRVEdataset%s.hd5'%suffix
    

    build_paramRVE(paramRVEname, ns, seed)

