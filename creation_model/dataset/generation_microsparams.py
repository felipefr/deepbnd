import sys, os
import numpy as np

from deepBND.__init__ import *
import deepBND.core.sampling.generation_inclusions as geni
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
    ns = 200 # training
    seed = 1 # for the test   
    paramRVEname = folder + 'paramRVEdataset_github.hd5'
    
    
    build_paramRVE(paramRVEname, ns, seed)

