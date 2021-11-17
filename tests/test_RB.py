import pytest
import sys, os
import numpy as np
from dolfin import *

import deepBND.core.data_manipulation.wrapper_h5py as myhd
from deepBND.core.fenics_tools.enriched_mesh import EnrichedMesh  

def test_computingBasis():
    from deepBND.creation_model.RB.generation_XYfiles import computingBasis
    folder = "./data/"
    
    suffix = '_github'
    nameSnaps = folder + 'snapshots%s.hd5'%suffix
    nameMeshRefBnd = folder + 'boundaryMesh.xdmf'
    nameWbasis = folder + 'Wbasis%s.hd5'%suffix
    nameYlist = folder + 'Y%s.hd5'%suffix
    nameXYlist = folder + 'XY%s.hd5'%suffix
    nameParamRVEdataset = folder + 'paramRVEdataset%s.hd5'%suffix
    
    Mref = EnrichedMesh(nameMeshRefBnd)
    Vref = VectorFunctionSpace(Mref,"CG", 1)
    
    dxRef = Measure('dx', Mref) 
    dsRef = Measure('ds', Mref) 
    Nh = Vref.dim()
    
    Nmax = 160
    
    computingBasis(nameSnaps, nameWbasis, Nmax, Nh, Vref, dsRef)
    
    
    Wbasis_A, Wbasis_S, M , sig_A, sig_S = myhd.loadhd5(nameWbasis, 
                        ['Wbasis_A', 'Wbasis_S', 'massMat', 'sig_A', 'sig_S'])
    
    assert Wbasis_A.shape[0] == Nmax and Wbasis_A.shape[1] == Nh
    assert Wbasis_S.shape[0] == Nmax and Wbasis_S.shape[1] == Nh
    assert np.allclose( Wbasis_A@M@Wbasis_A.T, np.eye(Nmax))
    assert np.allclose( Wbasis_S@M@Wbasis_S.T, np.eye(Nmax))
    assert np.allclose( np.linalg.norm(Wbasis_S[:,0]), 4.161791450043481 )
    assert np.allclose( np.linalg.norm(Wbasis_S[0,:]), 4.521092802358088 )
    assert np.allclose( np.linalg.norm(Wbasis_A[:,0]), 4.161791450051313 )
    assert np.allclose( np.linalg.norm(Wbasis_A[0,:]), 3.6052866881155223 )
    assert np.allclose( sig_A[0], 3.759670654566904)
    assert np.allclose( sig_S[0], 2.819559671161688)


# TODO : extractAlpha and createXY tests