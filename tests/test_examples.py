import sys
import pytest
import sys, os
import dolfin as df 
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
from mpi4py import MPI

import deepBND.__init__ import *
import deepBND.core.fenics_tools.misc as feut

def test_solve_cook():
    from deepBND.core.elasticity.fenics_utils import symgrad_voigt

    import deepBND.core.data_manipulation.wrapper_h5py as myhd
    
    from deepBND.examples.cook.mesh import CookMembrane
    from deepBND.examples.cook.cook import solve_cook
    
    Ny_split =  40 # 5, 10, 20, 40, 80
    caseType = 'reduced_per' # opt: reduced_per, dnn, full
    seed = 1
        
    # ========== dataset folders ================= 
    folder = rootDataPath + "/new_fe2/DNS/DNS_72_2/"
    folderMesh = rootDataPath + '/deepBND/cook/meshes_seed{0}/'.format(seed)
    
    tangentName = folder + 'tangents/tangent_%s.hd5'%caseType
    tangent_dataset = myhd.loadhd5(tangentName, 'tangent')
    ids = myhd.loadhd5(tangentName, 'id')
    center = myhd.loadhd5(tangentName, 'center')
    meshfile = folderMesh + 'mesh_%d.xdmf'%Ny_split
    
    lcar = 44.0/Ny_split  
    
    gmshMesh = CookMembrane(lcar = lcar)
    gmshMesh.write(savefile = meshfile, opt = 'fenics')
    
    np.random.seed(seed)
    
    uh, Chom = solve_cook(meshfile, tangent_dataset)
    
    mesh = uh.function_space().mesh()
    sigma = lambda u: df.dot(Chom, symgrad_voigt(u))
    
    
    IntegralDisp = np.array([-1808.96889191, 4707.28633332])
    IntegralStress = np.array([-6.62082936e-05, 3.13690525e+01, 3.83999954e+01])
    
    assert np.allclose( feut.Integral(uh, mesh.dx, shape=(2,)), IntegralDisp) 
    assert np.allclose( feut.Integral(sigma(uh), mesh.dx, shape=(3,)), IntegralStress)
    
    
def test_solve_DNS():
    from deepBND.examples.bar_DNS.solveDNS import solve_DNS
    
    # ========== dataset folders ================= 
    
    folder = rootDataPath + '/new_fe2/DNS/DNS_24/'
    FaceTraction = 3    
    ty = -0.01
    tx = 0.0    
    FacesClamped = [5]
        
    param = [FaceTraction, tx, ty, FacesClamped]
        
    uh = solve_DNS(folder + 'mesh.xdmf', param)
    
    mesh = uh.function_space().mesh()
    
    IntegralDisp = np.array([-1.33093136e-04, -3.38050779e-01])

    assert np.allclose( feut.Integral(uh, mesh.dx, shape=(2,)), IntegralDisp) 
    
    
def test_solve_barMultiscale():    
    from deepBND.core.elasticity.fenics_utils import symgrad_voigt
    from deepBND.examples.bar_multiscale.barMultiscale import solve_barMultiscale
    
    
    IntegralDisp = {'reduced_per': np.array([-7.98503130e-05, -3.31125648e-01]),
                    'full': np.array([-9.53547037e-05, -3.31426336e-01]),
                    'dnn_big': np.array([-9.60797298e-05, -3.31285703e-01])}
                                     
    IntegralStress = {'reduced_per': np.array([ 2.84999660e-08,  4.28544262e-06, -1.00000241e-02]),
                  'full': np.array([[ 1.6871474e-08, -1.2981735e-05, -1.0000028e-02]]),
                  'dnn_big': np.array([ 5.09102355e-08, -1.34413698e-05, -9.99989154e-03])}   
               

    Ny_DNS =  24 
    Lx = 2.0
    Ly = 0.5
    
    ty = -0.01
    tx = 0.0    
    
    param = [Lx, Ly, tx, ty]
    
    folder = rootDataPath + "/deepBND/DNS/DNS_%d_new/"%Ny_DNS
    meshfile = folder + "multiscale/meshBarMacro_Multiscale.xdmf"
    
    for caseType in ['reduced_per', 'dnn_big', 'full']:

        tangentName = folder + 'tangents/tangent_%s.hd5'%caseType
        
        uh, Chom = solve_barMultiscale(meshfile, tangentName, param)
    
        sigma = lambda u: df.dot(Chom, symgrad_voigt(u))
        
        mesh = uh.function_space().mesh()
        dx = df.Measure('dx', mesh)
        
        assert np.allclose( feut.Integral(uh, dx, shape=(2,)), IntegralDisp[caseType]) 
        assert np.allclose( feut.Integral(sigma(uh), dx, shape=(3,)), IntegralStress[caseType])



