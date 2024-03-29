
"""
This file is part of deepBND, a data-driven enhanced boundary condition implementaion for 
computational homogenization problems, using RB-ROM and Neural Networks.
Copyright (c) 2020-2023, Felipe Rocha.
See file LICENSE.txt for license information. 
Please cite this work according to README.md.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or <felipe.f.rocha@gmail.com>
"""

# Build paramRVEdataset.hd5: It contains geometrical description for each snapshot. It allows the mesh generation, etc. 

import sys, os
import numpy as np

from deepBND.__init__ import *
import deepBND.creation_model.dataset.generation_inclusions as geni
import deepBND.core.data_manipulation.wrapper_h5py as myhd
from deepBND.core.multiscale.mesh_RVE import paramRVE_default 

def build_paramRVE(paramRVEname, ns, seed):
    p = paramRVE_default()
    NR = p.Nx*p.Ny # 36
    
    # Radius Generation
    np.random.seed(seed)
    
    os.system('rm ' + paramRVEname)
    X, f = myhd.zeros_openFile(filename = paramRVEname,  shape = (ns,NR,5), label = 'param', mode = 'w')
    
    ellipseData_pattern = geni.getEllipse_emptyRadius(p.Nx,p.Ny,p.Lxt, p.Lyt, p.x0, p.y0)
    
    thetas = geni.getScikitoptSample(NR,ns, -1.0, 1.0,  seed, op = 'lhs')
    
    for i in range(ns):
        print("inserting on ", i)
        X[i,:,:] =  ellipseData_pattern 
        X[i,:,2] = geni.getRadiusExponential(p.r0, p.r1, thetas[i,:])
    
    
    f.close()


if __name__ == '__main__':
    
    folder = rootDataPath + "/deepBND/dataset/"
    ns = 20
    seed = 2    
    suffix = "_validation"
    paramRVEname = folder + 'paramRVEdataset%s.hd5'%suffix
    
    
    build_paramRVE(paramRVEname, ns, seed)

