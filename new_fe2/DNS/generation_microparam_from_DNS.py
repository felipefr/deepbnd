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


f = open("../../../rootDataPath.txt")
rootData = f.read()[:-1]
f.close()

folder = rootData + "/new_fe2/DNS/DNS_24/"

param_DNS = myhd.loadhd5(folder + 'param_DNS.hd5', 'param') 
fac = 4
Ny_DNS = 24
Nx_DNS = fac*Ny_DNS
Ly_DNS = 0.5
Lx_DNS = fac*Ly_DNS

H_DNS = Lx_DNS/Nx_DNS # same as Ly/Ny

# new
maxOffset = 2

H = 1.0 # size of each square
NxL = NyL = 2
NL = NxL*NyL
x0L = y0L = -H 
LxL = LyL = 2*H
Nx = (NxL+2*maxOffset)
Ny = (NyL+2*maxOffset)
Lxt = Nx*H
Lyt = Ny*H
r0 = 0.2*H
r1 = 0.4*H
x0 = -Lxt/2.0
y0 = -Lyt/2.0

ns = (Nx_DNS - Nx + 1)*(Ny_DNS - Ny + 1)

name_ellipseData_RVEs = folder + 'param_RVEs_from_DNS.hd5'

os.system('rm ' + name_ellipseData_RVEs)
X, f = myhd.zeros_openFile(name_ellipseData_RVEs,  [(ns,Ny*Nx,5),(ns,2)] , ['param','center'], mode = 'w-')
Xe , Xc = X


param_pattern = geni.getEllipse_emptyRadius(Nx,Ny,Lxt, Lyt, x0, y0)


R_DNS = param_DNS[:,2].reshape((Ny_DNS,Nx_DNS))
k = 0
facRadius_RVE_DNS = H/H_DNS


for j in range(0, Ny_DNS - Ny + 1):
    for i in range(0,Nx_DNS - Nx + 1):
        param = copy.deepcopy(param_pattern)
        param[:,2] = facRadius_RVE_DNS*R_DNS[j : j + Ny, i : i + Nx].flatten()
    
        Xe[k,:,:] = param
        Xc[k,0] = (i + maxOffset + 1)*H_DNS
        Xc[k,1] = (j + maxOffset + 1)*H_DNS

        k = k + 1 
                
f.close()

