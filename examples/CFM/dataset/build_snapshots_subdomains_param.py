#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 10:11:12 2022

@author: felipe
"""


import sys, os
import numpy as np
from timeit import default_timer as timer
import copy

from deepBND.__init__ import *
import deepBND.core.data_manipulation.wrapper_h5py as myhd
import deepBND.creation_model.dataset.generation_inclusions as geni
import deepBND.core.multiscale.mesh_RVE as mrve
from deepBND.core.mesh.ellipse_mesh_bar import ellipseMeshBar

def buildDNSmesh(Ny, paramfile, meshfile, readReuse, fac_lcar, seed):

    fac_x = 4
    x0 = y0 = 0.0
    Ly = 0.5
    Lx = fac_x*Ly
    Nx = fac_x*Ny
    H = Lx/Nx # same as Ly/Ny

    lcar = fac_lcar*H # 
        
    if(not readReuse):
        print("generating param") 
        
        r0 = 0.2*H
        r1 = 0.4*H

        NpLx = int(Lx/lcar) + 1 # affine boundary
        NpLy = int(Ly/lcar) + 1 # affine boundary
            
        np.random.seed(seed)
        param = geni.circularRegular2Regions(r0, r1, Nx, Ny, Lx, Ly, offset = 0, 
                                        ordered = False, x0 = x0, y0 = y0)[0]
        os.system('rm '+ paramfile)
        myhd.savehd5(paramfile, param, 'param', 'w-') 
    
    else:
        print("reading param")
        param = myhd.loadhd5(paramfile, label = 'param')
    
    
    meshGMSH = ellipseMeshBar(param, x0, y0, Lx, Ly, lcar = lcar)
    meshGMSH.write(meshfile, opt = 'fenics')


def buildRVEparam_fromDNS(Ny_DNS, paramDNSfile, paramRVEfile):    
    
    param_DNS = myhd.loadhd5(paramDNSfile, 'param')
     
    fac = 4
    Nx_DNS = fac*Ny_DNS
    Ly_DNS = 0.5
    Lx_DNS = fac*Ly_DNS
    
    H_DNS = Lx_DNS/Nx_DNS # same as Ly/Ny
    
    p = mrve.paramRVE_default()
    
    ns = (Nx_DNS - p.Nx + 1)*(Ny_DNS - p.Ny + 1)
    
    os.system('rm ' + paramRVEfile)
    X, f = myhd.zeros_openFile(paramRVEfile,  [(ns,),(ns,p.Ny*p.Nx,5),(ns,2)] , ['id', 'param','center'], mode = 'w-')
    Xid, Xe , Xc = X
    
    param_pattern = geni.getEllipse_emptyRadius(p.Nx,p.Ny,p.Lxt, p.Lyt, p.x0, p.y0)
    
    R_DNS = param_DNS[:,2].reshape((Ny_DNS,Nx_DNS)) # slow in y, fast in x
    k = 0
    facRadius_RVE_DNS = p.H/H_DNS
    
    for j in range(0, Ny_DNS - p.Ny + 1):
        for i in range(0,Nx_DNS - p.Nx + 1):
            param = copy.deepcopy(param_pattern)
            param[:,2] = facRadius_RVE_DNS*R_DNS[j : j + p.Ny, i : i + p.Nx].flatten()
        
            Xe[k,:,:] = param
            Xc[k,0] = (i + p.maxOffset + 1)*H_DNS
            Xc[k,1] = (j + p.maxOffset + 1)*H_DNS
            Xid[k] = k
            k = k + 1 
                    
    f.close()


if __name__ == '__main__':
    
    if(len(sys.argv)>1):
        Ny = int(sys.argv[1])
        readReuse = bool(sys.argv[2])
        export_paramRVE_fromDNS = bool(sys.argv[3])
        seed = int(sys.argv[4])
    else:
        Ny = 24
        readReuse = False
        export_paramRVE_fromDNS = True
        seed = 1 # seed is dummy in the case reuse is true

    folder = rootDataPath + "/bar_DNS/Ny_%d/"%Ny
    
    paramfile = folder + 'param_DNS.hd5'
    meshname = folder + 'mesh.xdmf'
    paramRVEfile = folder + 'paramRVEdataset.hd5'
    
    fac_lcar = 1/15 # or 1/9 more less the same order than the RVE
    
    buildDNSmesh(Ny, paramfile, meshname, readReuse, fac_lcar, seed)
    
    if(export_paramRVE_fromDNS):
        buildRVEparam_fromDNS(Ny, paramfile, paramRVEfile)   
    
