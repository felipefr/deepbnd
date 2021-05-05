import sys, os
sys.path.insert(0,'../../utils/')
import numpy as np
import myHDF5 as myhd
import meshUtils as meut
import generationInclusions as geni
from mpi4py import MPI

if(len(sys.argv)>1):
    run = int(sys.argv[1])
    num_runs = int(sys.argv[2])
else:
    comm = MPI.COMM_WORLD
    run = comm.Get_rank()
    num_runs = comm.Get_size()

print('run, num_runs ', run, num_runs)


f = open("../../../rootDataPath.txt")
rootData = f.read()[:-1]
f.close()

Ny_DNS = 24

folder = rootData + "/new_fe2/DNS/DNS_{0}_old/".format(Ny_DNS)
folderMesh = folder + 'meshes/'
nameParamRVEdata = folder + 'ellipseData_RVEs.hd5'

def write_mesh(paramRVEdata, nameMesh, isOrdenated = False, size = 'reduced'):

    p = geni.paramRVE() # load default parameters
    
    if(isOrdenated):
        permTotal = np.arange(0,p.Nx*p.Ny).astype('int')
    else:
        permTotal = geni.orderedIndexesTotal(p.Nx,p.Ny,p.NxL)
    
    paramRVEdata[:,:] = paramRVEdata[permTotal,:]

    if(size == 'reduced'):
        meshGMSH = meut.ellipseMesh2(paramRVEdata[:4,:], p.x0L, p.y0L, p.LxL, p.LyL, p.lcar) # it should be choosen adequate four
        meshGMSH.setTransfiniteBoundary(p.NpLxL)
        
    elif(size == 'full'):
        meshGMSH = meut.ellipseMesh2Domains(p.x0L, p.y0L, p.LxL, p.LyL, p.NL, paramRVEdata, 
                                            p.Lxt, p.Lyt, p.lcar, x0 = p.x0, y0 = p.y0)
        meshGMSH.setTransfiniteBoundary(p.NpLxt)
        meshGMSH.setTransfiniteInternalBoundary(p.NpLxL)   
            
    meshGMSH.write(nameMesh, opt = 'fenics')
    
paramRVEdata = myhd.loadhd5(nameParamRVEdata, 'ellipseData')[run::num_runs,:,:]
nrun = len(paramRVEdata)

ids = np.array([run + i*num_runs for i in range(nrun)])

for i in range(nrun):
    print("gererated ", i , ids[i], " mesh")
    write_mesh(paramRVEdata[i,:,:], folderMesh + 'mesh_micro_{0}_reduced.xdmf'.format(ids[i]), isOrdenated= True, size = 'reduced')