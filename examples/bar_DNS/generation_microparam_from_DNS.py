import sys, os
sys.path.insert(0,'../../utils/')
import numpy as np
import matplotlib.pyplot as plt
import fenicsMultiscale as fmts
import myHDF5 as myhd
import meshUtils as meut
import generationInclusions as geni
from timeit import default_timer as timer
import copy

# i : varies on horizontal
# j : varies on vertical
# id : row (slow index), column (fast index)

f = open("../../../rootDataPath.txt")
rootData = f.read()[:-1]
f.close()

folder = rootData + "/new_fe2/DNS/DNS_72_old/"

# param_DNS = myhd.loadhd5(folder + 'param_DNS.hd5', 'param') 
param_DNS = myhd.loadhd5(folder + 'param_DNS.hd5', 'ellipseData')
 
fac = 4
Ny_DNS = 72
Nx_DNS = fac*Ny_DNS
Ly_DNS = 0.5
Lx_DNS = fac*Ly_DNS

H_DNS = Lx_DNS/Nx_DNS # same as Ly/Ny

# new
p = geni.paramRVE()

ns = (Nx_DNS - p.Nx + 1)*(Ny_DNS - p.Ny + 1)

nameParamRVEs = folder + 'param_RVEs_from_DNS.hd5'

os.system('rm ' + nameParamRVEs)
X, f = myhd.zeros_openFile(nameParamRVEs,  [(ns,),(ns,p.Ny*p.Nx,5),(ns,2)] , ['id', 'param','center'], mode = 'w-')
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

