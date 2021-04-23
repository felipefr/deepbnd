import sys, os
sys.path.insert(0,'../../utils/')
import numpy as np
import myHDF5 as myhd
import meshUtils as meut
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_ranks = comm.Get_size()

print('rank, num_ranks ', rank, num_ranks)

def write_mesh(ellipseData, nameMesh, size = 'reduced'):
    maxOffset = 2
    
    H = 1.0 # size of each square
    NxL = NyL = 2
    NL = NxL*NyL
    x0L = y0L = -H 
    LxL = LyL = 2*H
    # lcar = (2/30)*H
    lcar = (2/30)*H
    Nx = (NxL+2*maxOffset)
    Ny = (NyL+2*maxOffset)
    Lxt = Nx*H
    Lyt = Ny*H
    NpLxt = int(Lxt/lcar) + 1
    NpLxL = int(LxL/lcar) + 1
    print("NpLxL=", NpLxL) 
    x0 = -Lxt/2.0
    y0 = -Lyt/2.0
    r0 = 0.2*H
    r1 = 0.4*H
    Vfrac = 0.282743
    rm = H*np.sqrt(Vfrac/np.pi)

    if(size == 'reduced'):
        meshGMSH = meut.ellipseMesh2(ellipseData[:4,:], x0L, y0L, LxL, LyL, lcar)
        meshGMSH.setTransfiniteBoundary(NpLxL)
        
    elif(size == 'full'):
        meshGMSH = meut.ellipseMesh2Domains(x0L, y0L, LxL, LyL, NL, ellipseData[:36,:], Lxt, Lyt, lcar, x0 = x0, y0 = y0)
        meshGMSH.setTransfiniteBoundary(NpLxt)
        meshGMSH.setTransfiniteInternalBoundary(NpLxL)   
            
    meshGMSH.write(nameMesh, opt = 'fenics')
    
# loading the DNN model
ellipseData = myhd.loadhd5('ellipseData_RVEs.hd5', 'ellipseData')

# defining the micro model
i = 0
nCells = len(ellipseData)

for i in range(nCells):
    if(i%num_ranks == rank):
        write_mesh(ellipseData[i], './meshes/mesh_micro_{0}_reduced.xdmf'.format(i), 'reduced')