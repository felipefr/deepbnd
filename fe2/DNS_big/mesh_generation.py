import sys, os
sys.path.insert(0,'../../utils/')
import numpy as np
import matplotlib.pyplot as plt
import fenicsMultiscale as fmts
import myHDF5 as myhd
import meshUtils as meut
import generationInclusions as geni
from timeit import default_timer as timer


def enforceVfrac_indexes(R, indexes, H, Vfrac): # radius should be ordened interior to exterior, 
    alphaFrac = H*np.sqrt(len(indexes)*Vfrac/(np.pi*np.sum(R[indexes]**2)))
    R[indexes] *= alphaFrac
    
def get_indexes_window(Nx, Ny, Nx_w, Ny_w, Nx_0, Ny_0):
    indexes = np.arange(Nx*Ny).flatten().astype('int')
    return indexes.reshape((Ny,Nx))[Ny_0:Ny_0 + Ny_w,Nx_0:Nx_0 + Nx_w].flatten()

def get_indexes_ring(Nx, Ny, Nx_w, Ny_w, Nx_0, Ny_0):
    foo = lambda i: get_indexes_window(Nx, Ny, Nx_w-2*i, Ny_w-2*i, Nx_0+i, Ny_0+i)
    return np.array(list(set(foo(0)) - set(foo(1)))) 

fac = 4
Ly = 0.5
Lx = fac*Ly
Ny = 6
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
Nx_W = Ny_W = 6
Nx_w = Ny_w = 2


for i in range(0,Nx - Nx_W + 1, Nx_W):
    for j in range(0, Ny - Ny_W + 1, Ny_W):
        enforceVfrac_indexes(ellipseData[:,2], get_indexes_window(Nx,Ny,Nx_w,Ny_w,i+2,j+2), H, Vfrac)
        enforceVfrac_indexes(ellipseData[:,2], get_indexes_ring(Nx,Ny,Nx_W,Ny_W,i,j), H, Vfrac)
        enforceVfrac_indexes(ellipseData[:,2], get_indexes_ring(Nx,Ny,Nx_W-2,Ny_W-2,i+1,j+1), H, Vfrac)


print(np.pi*np.sum(ellipseData[:,2]**2))
print(np.max(ellipseData[:,2]), np.min(ellipseData[:,2]), r1, r0)


meshGMSH = meut.ellipseMeshBarAdaptative(ellipseData, x0, y0, Lx, Ly, lcar = [lcar,lcar,lcar])
meshGMSH.setTransfiniteBoundary(NpLx, direction = 'horiz')
meshGMSH.setTransfiniteBoundary(NpLy, direction = 'vert')
# meshGMSH.addMeshConstraints()
# meshGMSH.writeGeo('DNS.geo')
meshGMSH.write('./DNS_{0}/mesh.xdmf'.format(Ny), opt = 'fenics')
os.system('rm ' + './DNS_{0}/ellipseData_DNS.hd5'.format(Ny))
myhd.savehd5('./DNS_{0}/ellipseData_DNS.hd5'.format(Ny), ellipseData, 'ellipseData', 'w-') 

