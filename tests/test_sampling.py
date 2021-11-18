import pytest
import sys, os
import numpy as np

import deepBND.core.data_manipulation.wrapper_h5py as myhd

def test_build_paramRVE():
    import deepBND.creation_model.dataset.generation_inclusions as geni
    from deepBND.core.multiscale.mesh_RVE import paramRVE_default
    from deepBND.creation_model.dataset.build_snapshots_param import build_paramRVE

    ns = 512 
    seed = 13   
    paramRVEname = 'paramRVEdataset_temp.hd5'
    
    build_paramRVE(paramRVEname, ns, seed)
    
    paramRVE = myhd.loadhd5(paramRVEname, 'param')
    
    i1 = np.random.randint(0,ns)
    i2 = np.random.randint(0,ns)
    
    assert np.allclose( paramRVE[i1,:,0], paramRVE[i2,:,0]) 
    assert np.allclose( paramRVE[i1,:,1], paramRVE[i2,:,1]) 
    assert np.allclose( np.sum(paramRVE[:,:,3], axis=0), ns*np.ones(36)) # this column should be 1.0
    assert np.allclose( np.sum(paramRVE[:,:,4], axis=0), np.zeros(36)) # this column should be 0.0
    assert np.allclose( np.sum(paramRVE[:,:,2]), 5318.350592592172 )
    assert np.allclose( np.std(paramRVE[:,:,2]), 0.05750590828174954 )

    os.system('rm ' + paramRVEname)


test_build_paramRVE()