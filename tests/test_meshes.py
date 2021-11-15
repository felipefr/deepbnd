import pytest
import sys, os
import dolfin as df 
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
from mpi4py import MPI

sys.path.insert(0,'../')
import core.fenics_tools.misc as feut

def test_buildDNSmesh():
    import core.data_manipulation.wrapper_h5py as myhd    
    from examples.bar_DNS.mesh_generation_DNS import buildDNSmesh
    from core.fenics_tools.enriched_mesh import EnrichedMesh  

    Ny = 12
    readParam = False
            
    paramfile = 'param_DNS_temp.hd5'
    meshfile = 'mesh_temp.xdmf'
    
    fac_lcar = 0.15 # or 1/9 more less the same order than the RVE
    seed = 1
    
    buildDNSmesh(Ny, paramfile, meshfile, readParam, fac_lcar, seed)

    mesh = EnrichedMesh(meshfile)    
    
    vol0 = feut.Integral([df.Constant(1.0)], mesh.dx(0), shape=(1,))[0]
    vol1 = feut.Integral([df.Constant(1.0)], mesh.dx(1), shape=(1,))[0]
     
    volTot_target = 1.0000000000000018
    volRatio_target = 0.2663513594794764
    
    assert np.allclose( np.array([vol1 + vol0 , vol0/(vol1 + vol0)]), 
                        np.array([volTot_target, volRatio_target]) ) 

    os.system('rm ' + paramfile)
    os.system('rm ' + meshfile[:-5] + '*')
    