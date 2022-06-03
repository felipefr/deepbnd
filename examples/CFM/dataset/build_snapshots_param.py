#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 03:41:50 2022

@author: felipe
"""

# Build paramRVEdataset.hd5: It contains geometrical description for each snapshot. It allows the mesh generation, etc. 

import sys, os
import numpy as np
import copy

from deepBND.__init__ import *
import deepBND.creation_model.dataset.generation_inclusions as geni
import deepBND.core.data_manipulation.wrapper_h5py as myhd
from deepBND.core.multiscale.mesh_RVE import paramRVE_default 

def build_paramRVE_subdomains(paramRVEname, Nx, Ny, maxOffset):    
    
    paramRVE = myhd.loadhd5(paramRVEname, 'param')
     
    p = paramRVE_default(Nx - 4*maxOffset, Ny - 4*maxOffset, maxOffset = maxOffset)
    
    Nx_subs = int( (Nx - 2*maxOffset)/maxOffset )
    Ny_subs = int( (Ny - 2*maxOffset)/maxOffset )
    
    ns = len(paramRVE)* Nx_subs * Ny_subs  # gives 9, but maybe does not generalise
    
    paramRVE_subdomains_file = paramRVEname.split('.')[0] + "_subdomains.hd5"
    
    os.system('rm ' + paramRVE_subdomains_file)
    X, f = myhd.zeros_openFile(paramRVE_subdomains_file,  [(ns,),(ns,), (ns,p.Ny*p.Nx,5), (ns,2)] , 
                               ['id', 'id_local', 'param', 'center'], mode = 'w-')
    
    id_ , id_local, param_subdomains, center = X
    
    param_pattern = geni.getEllipse_emptyRadius(p.Nx,p.Ny,p.Lxt, p.Lyt, p.x0, p.y0)
    
    
    kk = 0
    for k in range(len(paramRVE)):
        Radii = paramRVE[k,:,2].reshape((Ny,Nx)) # slow in y, fast in x
    
        for j in range(Ny_subs):
            for i in range(Nx_subs):
                param_subdomain_ij = copy.deepcopy(param_pattern)
                param_subdomain_ij[:,2] = Radii[j : j + p.Ny, i : i + p.Nx].flatten()
            
                id_[kk] = k
                id_local[kk] = j*Nx_subs + i
                param_subdomains[kk,:,:] = param_subdomain_ij
                center[kk, 0] = p.x0 + (i + p.maxOffset)*p.H
                center[kk, 1] = p.y0 + (j + p.maxOffset)*p.H
                
                kk = kk + 1 
                    
    f.close()



def build_paramRVE(paramRVEname, ns, seed):
    p = paramRVE_default(NxL = 6, NyL = 6, maxOffset = 2)
    NR = p.Nx*p.Ny 
    
    # Radius Generation
    np.random.seed(seed)
    
    os.system('rm ' + paramRVEname)
    X, f = myhd.zeros_openFile(filename = paramRVEname,  shape = (ns,NR,5), label = 'param', mode = 'w')
    
    ellipseData_pattern = geni.getEllipse_emptyRadius(p.Nx,p.Ny,p.Lxt, p.Lyt, p.x0, p.y0)
    
    
    thetas = geni.getScikitoptSample(NR,ns, -1.0, 1.0,  seed, op = 'lhs')
    
    for i in range(ns):
        print("inserting on ", i)
        X[i,:,:] = ellipseData_pattern
        X[i,:,2] = geni.getRadiusExponential(p.r0, p.r1, thetas[i,:]) # to give the same area
        
    f.close()
    


if __name__ == '__main__':
    
    folder = rootDataPath + "/CFM/dataset/"
    ns = 100
    seed = 1 
    suffix = "_ns100"
    paramRVEname = folder + 'paramRVEdataset%s.hd5'%suffix
    

    build_paramRVE(paramRVEname, ns, seed)
    build_paramRVE_subdomains(paramRVEname, Nx = 10, Ny = 10, maxOffset = 2)
