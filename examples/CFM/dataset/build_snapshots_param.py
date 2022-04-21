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
import deepBND.core.data_manipulation.wrapper_h5py as myhd
from deepBND.core.multiscale.mesh_RVE import paramRVE_default 

def build_paramRVE(paramRVEname, ns, seed, excentricity = 1.0):
    p = paramRVE_default(NxL = 6, NyL = 6, maxOffset = 2)
    NR = p.Nx*p.Ny 
    
    # Radius Generation
    # np.random.seed(seed)
    
    # os.system('rm ' + paramRVEname)
    # X, f = myhd.zeros_openFile(filename = paramRVEname,  shape = (ns,NR,5), label = 'param', mode = 'w')
    
    # ellipseData_pattern = geni.getEllipse_emptyRadius(p.Nx,p.Ny,p.Lxt, p.Lyt, p.x0, p.y0)
    
    
    # thetas = geni.getScikitoptSample(NR,ns, -np.pi, np.pi,  seed, op = 'lhs')
    
    # for i in range(ns):
    #     print("inserting on ", i)
    #     X[i,:,:] = ellipseData_pattern
    #     X[i,:,3] = excentricity
    #     X[i,:,2] = np.sqrt(2)*p.rm # to give the same area
    #     X[i,:,4] = thetas[i,:]
            
    # f.close()
    
    return p


if __name__ == '__main__':
    
    folder = rootDataPath + "/CFM/dataset/"
    ns = 2
    seed = 2    
    suffix = ""
    paramRVEname = folder + 'paramRVEdataset%s.hd5'%suffix
    

    p = build_paramRVE(paramRVEname, ns, seed, excentricity = 1.0)

