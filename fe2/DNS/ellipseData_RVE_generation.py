import sys, os
sys.path.insert(0,'../../utils/')
import numpy as np
import matplotlib.pyplot as plt
import fenicsMultiscale as fmts
import myHDF5 as myhd
import meshUtils as meut
import generationInclusions as geni
from timeit import default_timer as timer


def enforceVfracPerOffset(radius, NxL, maxOffset, H, Vfrac): # radius should be ordened interior to exterior, 
    for i in range(maxOffset+1):
        ni =  (NxL + 2*(i-1))**2 
        nout = (NxL + 2*i)**2
        alphaFrac = H*np.sqrt((nout-ni)*Vfrac/(np.pi*np.sum(radius[ni:nout]**2)))
        radius[ni:nout] *= alphaFrac
        
    return radius


ellipseData_DNS = myhd.loadhd5('ellipseData_DNS.hd5', 'ellipseData') 
fac = 4
Ny_DNS = 7
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
Vfrac = 0.282743

ns = (Nx_DNS - Nx + 1)*(Ny_DNS - Ny + 1)

enforceVolFrac = True
name_ellipseData_RVEs = 'ellipseData_RVEs_volFrac.hd5' if enforceVolFrac else 'ellipseData_RVEs.hd5'

os.system('rm ' + name_ellipseData_RVEs)
X, f = myhd.zeros_openFile(name_ellipseData_RVEs,  [(ns,Ny*Nx,5),(ns,2)] , ['ellipseData','center'], mode = 'w-')
Xe , Xc = X


ellipseData, PermTotal, dummy = geni.circularRegular2Regions(r0, r1, NxL, NyL, Lxt, Lyt, 
                                                             maxOffset, ordered = False, x0 = x0, y0 = y0) # just a dummy creation

R_DNS = ellipseData_DNS[:,2].reshape((Ny_DNS,Nx_DNS))
k = 0


for j in range(0, Ny_DNS - Ny + 1):
    for i in range(0,Nx_DNS - Nx + 1):
        ellipseData[:,2] = R_DNS[j : j + Ny, i : i + Nx].flatten()
        ellipseData = ellipseData[PermTotal]
        
        if(enforceVolFrac):
            ellipseData[:,2] = enforceVfracPerOffset(ellipseData[:,2], NxL, maxOffset, H, Vfrac) 
        else:
            facRadius_RVE_DNS = H/H_DNS
            ellipseData[:,2] *= facRadius_RVE_DNS
        
        
        Xe[k,:,:] = ellipseData
        Xc[k,0] = (i + maxOffset + 1)*H_DNS
        Xc[k,1] = (j + maxOffset + 1)*H_DNS


        k = k + 1 
                
f.close()

