import sys, os
sys.path.insert(0,'../../utils/')
import numpy as np
import matplotlib.pyplot as plt
import fenicsMultiscale as fmts
import myHDF5 as myhd
import meshUtils as meut
import generationInclusions as geni
from timeit import default_timer as timer


def enforceVfrac(R, Nx, Ny, Nx_w, Ny_w, Nx_0, Ny_0, H, Vfrac): # radius should be ordened interior to exterior, 
    R_ = R.reshape((Ny,Nx)) # stack vertically + C conventions
    alphaFrac = H*np.sqrt(Nx_w*Ny_w*Vfrac/(np.pi*np.sum(R_[Ny_0:Ny_0 + Ny_w+1,Nx_0:Nx_0 + Nx_w+1].flatten()**2)))
    # print(alphaFrac)
    R_[Ny_0:Ny_0 + Ny_w + 1,Nx_0:Nx_0 + Nx_w + 1] *= alphaFrac
    return R_.flatten()

fac = 4
Ly = 0.5
Lx = fac*Ly
Ny = 7
Nx = fac*Ny
H = Lx/Nx # same as Ly/Ny
x0 = y0 = 0.0
lcar = (2/30)*H # more less the same order than the RVE
r0 = 0.2*H
r1 = 0.4*H
Vfrac = 0.282743
rm = H*np.sqrt(Vfrac/np.pi)

NpLx = int(Lx/lcar) + 1 # affine boundary
NpLy = int(Ly/lcar) + 1 # affine boundary

np.random.seed(1)
ellipseData = geni.circularRegular2Regions(r0, r1, Nx, Ny, Lx, Ly, offset = 0, ordered = False, x0 = x0, y0 = y0)[0]
Nx_w = Ny_w = 6

print(np.max(ellipseData[:,2]), np.min(ellipseData[:,2]))

for i in range(0,Nx - Nx_w + 1):
    for j in range(0, Ny - Ny_w + 1):
        ellipseData[:,2] = enforceVfrac(ellipseData[:,2], Nx, Ny, Nx_w, Ny_w, i, j, H, Vfrac)


ellipseData[ellipseData[:,2]>r1,2] = r1
ellipseData[ellipseData[:,2]<r0,2] = r0

ellipseData[:,2] = enforceVfrac(ellipseData[:,2], Nx, Ny, Nx, Ny, 0, 0, H, Vfrac)

print(np.max(ellipseData[:,2]), np.min(ellipseData[:,2]), r1, r0)

meshGMSH = meut.ellipseMeshBarAdaptative(ellipseData, x0, y0, Lx, Ly, lcar = [lcar,lcar,lcar])
meshGMSH.setTransfiniteBoundary(NpLx, direction = 'horiz')
meshGMSH.setTransfiniteBoundary(NpLy, direction = 'vert')
# meshGMSH.addMeshConstraints()
# meshGMSH.writeGeo('DNS.geo')
meshGMSH.write('DNS.xdmf', opt = 'fenics')
os.system('rm ellipseData_DNS.hd5')
myhd.savehd5('ellipseData_DNS.hd5', ellipseData, 'ellipseData', 'w-') 

