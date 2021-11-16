import sys, os
import numpy as np
import deepBND.core.data_manipulation.wrapper_h5py as myhd
from deepBND.core.fenics_tools.enriched_mesh import EnrichedMesh 
from deepBND.core.multiscale.mesh_RVE import buildRVEmesh
from mpi4py import MPI

from deepBND.__init__ import *

def buildMeshRVEs_fromDNS(run, num_runs, nameParamRVEdata, nameMesh, size = 'reduced'):
    paramRVEdata = myhd.loadhd5(nameParamRVEdata, 'param')[run::num_runs,:,:]
    nrun = len(paramRVEdata)
    
    ids = np.array([run + i*num_runs for i in range(nrun)])
    
    for i in range(nrun):
        print("gererated ", i , ids[i], " mesh")
            print(paramRVEdata[i,:,:])
        buildRVEmesh(paramRVEdata[i,:,:], nameMesh.format(ids[i]), isOrdenated = False, size = size)


if __name__ == '__main__':
    
    if(len(sys.argv)>1):
        Ny_DNS = int(sys.argv[1])
        run = int(sys.argv[2])
        num_runs = int(sys.argv[3])
    else:
        Ny_DNS = 24
        comm = MPI.COMM_WORLD
        run = comm.Get_rank()
        num_runs = comm.Get_size()

    print('run, num_runs ', run, num_runs)
    
    folder = rootDataPath + "/deepBND/bar_DNS/Ny_{0}/".format(Ny_DNS)
    folderMesh = folder + 'meshes/'
    nameParamRVEdata = folder + 'param_RVEs_from_DNS.hd5'
    nameMesh = folderMesh + 'mesh_micro_{0}_reduced.xdmf'


    buildMeshRVEs_fromDNS(run, num_runs, nameParamRVEdata, nameMesh, size = 'reduced')

    
