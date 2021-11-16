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



def test_buildRVEmesh():
    
    from core.multiscale.mesh_RVE import buildRVEmesh
    from core.fenics_tools.enriched_mesh import EnrichedMesh  
    
    meshname = 'mesh_{0}_temp.xdmf'
    vol_targets = {'reduced' : [4.000000000000001, 0.22481976337016096], 
                   'full' : [ 36.00000000000007, 0.29270474461987417] }
    
    for label in ['reduced', 'full']:
    
        paramRVEdata = np.array([[-2.5,  -2.5,   0.32365094, 1., 0.],
                                 [-1.5,  -2.5,   0.3542752 , 1., 0.],
                                 [-0.5,  -2.5,   0.27619141, 1., 0.],
                                 [ 0.5,  -2.5,   0.38384515, 1., 0.],
                                 [ 1.5,  -2.5,   0.34912406, 1., 0.],
                                 [ 2.5,  -2.5,   0.39541271, 1., 0.],
                                 [-2.5,  -1.5,   0.28708686, 1., 0.],
                                 [-1.5,  -1.5,   0.28397141, 1., 0.],
                                 [-0.5,  -1.5,   0.3513896 , 1., 0.],
                                 [ 0.5,  -1.5,   0.21040868, 1., 0.],
                                 [ 1.5,  -1.5,   0.35707002, 1., 0.],
                                 [ 2.5,  -1.5,   0.34924407, 1., 0.],
                                 [-2.5,  -0.5,   0.31925386, 1., 0.],
                                 [-1.5,  -0.5,   0.21127192, 1., 0.],
                                 [-0.5,  -0.5,   0.20458835, 1., 0.],
                                 [ 0.5,  -0.5,   0.25080665, 1., 0.],
                                 [ 1.5,  -0.5,   0.26146398, 1., 0.],
                                 [ 2.5,  -0.5,   0.29260762, 1., 0.],
                                 [-2.5,   0.5,   0.39602428, 1., 0.],
                                 [-1.5,   0.5,   0.23842982, 1., 0.],
                                 [-0.5,   0.5,   0.33760571, 1., 0.],
                                 [ 0.5,   0.5,   0.26501235, 1., 0.],
                                 [ 1.5,   0.5,   0.24884199, 1., 0.],
                                 [ 2.5,   0.5,   0.34117594, 1., 0.],
                                 [-2.5,   1.5,   0.21508203, 1., 0.],
                                 [-1.5,   1.5,   0.23969786, 1., 0.],
                                 [-0.5,   1.5,   0.27800727, 1., 0.],
                                 [ 0.5,   1.5,   0.22991412, 1., 0.],
                                 [ 1.5,   1.5,   0.38176451, 1., 0.],
                                 [ 2.5,   1.5,   0.32857999, 1., 0.],
                                 [-2.5,   2.5,   0.29505946, 1., 0.],
                                 [-1.5,   2.5,   0.26321558, 1., 0.],
                                 [-0.5,   2.5,   0.39465668, 1., 0.],
                                 [ 0.5,   2.5,   0.32837373, 1., 0.],
                                 [ 1.5,   2.5,   0.35843306, 1., 0.],
                                 [ 2.5,   2.5,   0.21847496, 1., 0.]])
        

        buildRVEmesh(paramRVEdata, meshname.format(label), 
                                     isOrdinated = False, size = label)
        
        mesh = EnrichedMesh(meshname.format(label)) 
        
        vol0 = feut.Integral([df.Constant(1.0)], [mesh.dx(0), mesh.dx(2)], shape=(1,))[0]
        vol1 = feut.Integral([df.Constant(1.0)], [mesh.dx(1), mesh.dx(3)], shape=(1,))[0]

        assert np.allclose( np.array([vol1 + vol0 , vol0/(vol1 + vol0)]), 
                            np.array(vol_targets[label]) ) 
    
        os.system('rm ' [:-5] + '*')
