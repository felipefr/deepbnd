import sys, os
sys.path.insert(0,'../../utils/')
import numpy as np
import myHDF5 as myhd
import meshUtils as meut
import generationInclusions as geni
from timeit import default_timer as timer
import ioFenicsWrappers as iofe

f = open("../../../rootDataPath.txt")
rootData = f.read()[:-1]
f.close()

# Ny = int(input("type Ny : "))
# readParam = bool(input("read param? : "))

Ny = 72
readParam = True

folder = rootData + "/new_fe2/DNS/DNS_%d_2/"%Ny

fac = 4
Ly = 0.5
Lx = fac*Ly
Nx = fac*Ny
H = Lx/Nx # same as Ly/Ny
x0 = y0 = 0.0
# lcar = (1/9)*H # more less the same order than the RVE
lcar = (1/5)*H # more less the same order than the RVE

r0 = 0.2*H
r1 = 0.4*H
he = 0.075*H

NpLx = int(Lx/lcar) + 1 # affine boundary
NpLy = int(Ly/lcar) + 1 # affine boundary


if(not readParam):
    print("generating param") 
    np.random.seed(9)
    param = geni.circularRegular2Regions(r0, r1, Nx, Ny, Lx, Ly, offset = 0, ordered = False, x0 = x0, y0 = y0)[0]
    os.system('rm ' + folder + 'param_DNS.hd5')
    myhd.savehd5(folder + 'param_DNS.hd5', param, 'param', 'w-') 

else:
    print("reading param")
    param = myhd.loadhd5(folder + 'param_DNS.hd5', label = 'param')


# meshGMSH = meut.ellipseMeshBarAdaptative_3circles(param, x0, y0, Lx, Ly, lcar = [lcar,0.35*lcar,1.2*lcar], he = [he,he])
meshGMSH = meut.ellipseMeshBar(param, x0, y0, Lx, Ly, lcar = lcar)
# meshGMSH = meut.ellipseMeshBarAdaptative(param, x0, y0, Lx, Ly , lcar = 3*[lcar])
# meshGMSH.setTransfiniteBoundary(NpLx, direction = 'horiz')
# meshGMSH.setTransfiniteBoundary(NpLy, direction = 'vert')
# meshGMSH.addMeshConstraints()

# meshGMSH.writeGeo(folder + 'DNS.geo')
meshGMSH.write(folder + 'mesh.xdmf', opt = 'fenics')


# os.system("gmsh %s -2 -format 'msh2' -o %s"%(folder + 'DNS.geo', folder + 'DNS.msh' ))
# iofe.exportMeshXDMF_fromGMSH(folder + 'DNS.msh',  folder + 'mesh.xdmf')


