import sys
import pytest
import sys, os
import dolfin as df 
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
from mpi4py import MPI

sys.path.insert(0,'../')

def test_solve_cook():
    from core.elasticity.fenics_utils import symgrad_voigt
    import core.fenics_tools.misc as feut
    import core.data_manipulation.wrapper_h5py as myhd
    
    from examples.cook.mesh import CookMembrane
    from examples.cook.cook import solve_cook
    
    Ny_split =  40 # 5, 10, 20, 40, 80
    caseType = 'reduced_per' # opt: reduced_per, dnn, full
    seed = 1
        
    # ========== dataset folders ================= 
    rootDataPath = open('../rootDataPath.txt','r').readline()[:-1]
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
    import core.fenics_tools.misc as feut
    from examples.DNS.solveDNS import solve_DNS
    

    # ========== dataset folders ================= 
    rootDataPath = open('../rootDataPath.txt','r').readline()[:-1]
    
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
    
