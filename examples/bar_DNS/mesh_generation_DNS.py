import sys, os
sys.path.insert(0,'../..')
import numpy as np
from timeit import default_timer as timer
import copy

import core.data_manipulation.wrapper_h5py as myhd
import core.sampling.generation_inclusions as geni
from core.mesh.ellipse_mesh_bar import ellipseMeshBar

def buildDNSmesh(Ny, paramfile, meshfile, readParam, fac_lcar, seed):

    fac_x = 4
    x0 = y0 = 0.0
    Ly = 0.5
    Lx = fac_x*Ly
    Nx = fac_x*Ny
    H = Lx/Nx # same as Ly/Ny

    lcar = fac_lcar*H # 
        
    if(not readParam):
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
    
    p = geni.paramRVE()
    
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
        readParam = bool(sys.argv[2])
        export_paramRVE_fromDNS = bool(sys.argv[3])
    else:
        Ny = 24
        readParam = False
        export_paramRVE_fromDNS = True
        
    rootDataPath = open('../../rootDataPath.txt','r').readline()[:-1]

    folder = rootDataPath + "/deepBND/bar_DNS/Ny_%d/"%Ny
    
    paramfile = folder + 'param_DNS.hd5'
    meshfile = folder + 'mesh.xdmf'
    
    fac_lcar = 1/5 # or 1/9 more less the same order than the RVE
    seed = 9
    
    buildDNSmesh(Ny, paramfile, meshfile, readParam, fac_lcar, seed)
    
    if(export_paramRVE_fromDNS):
        paramRVEfile = folder + 'param_RVEs_from_DNS.hd5'
        buildRVEparam_fromDNS(Ny, paramfile, paramRVEfile)   
    
