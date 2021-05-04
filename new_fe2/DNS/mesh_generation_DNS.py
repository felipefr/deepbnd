import sys, os
sys.path.insert(0,'../../utils/')
import numpy as np
import myHDF5 as myhd
import meshUtils as meut
import generationInclusions as geni
from timeit import default_timer as timer

f = open("../../../rootDataPath.txt")
rootData = f.read()[:-1]
f.close()

folder = rootData + "/new_fe2/DNS/"

fac = 4
Ly = 0.5
Lx = fac*Ly
Ny = int(input("type Ny : "))
Nx = fac*Ny
H = Lx/Nx # same as Ly/Ny
x0 = y0 = 0.0
lcar = (1/9)*H # more less the same order than the RVE
r0 = 0.2*H
r1 = 0.4*H
he = 0.075*H

NpLx = int(Lx/lcar) + 1 # affine boundary
NpLy = int(Ly/lcar) + 1 # affine boundary

np.random.seed(8)
param = geni.circularRegular2Regions(r0, r1, Nx, Ny, Lx, Ly, offset = 0, ordered = False, x0 = x0, y0 = y0)[0]
Nx_W = Ny_W = 6
Nx_w = Ny_w = 2

# meshGMSH = meut.ellipseMeshBarAdaptative_3circles(ellipseData, x0, y0, Lx, Ly, lcar = [lcar,0.35*lcar,lcar], he = [he,he])
# meshGMSH.setTransfiniteBoundary(NpLx, direction = 'horiz')
# meshGMSH.setTransfiniteBoundary(NpLy, direction = 'vert')
# # meshGMSH.addMeshConstraints()
# # meshGMSH.writeGeo('./DNS_{0}_light/DNS.geo')
# meshGMSH.write('./DNS_{0}_light/mesh.xdmf'.format(Ny), opt = 'fenics')
os.system('rm ' + folder + '/DNS_{0}/param_DNS.hd5'.format(Ny))
myhd.savehd5(folder + '/DNS_{0}/param_DNS.hd5'.format(Ny), param, 'param', 'w-') 

