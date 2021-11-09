import sys, os
import dolfin as df 
import matplotlib.pyplot as plt
import numpy as np
from ufl import nabla_div
from timeit import default_timer as timer
import multiphenics as mp
from mpi4py import MPI

sys.path.insert(0,'../../')
from core.fenics.utils import symgrad, symgrad_voigt
import core.fenics.utils as feut
import core.multiscale.fenics_multiscale as fmts
import core.data_manipulation.my_hdf5 as myhd
from core.fenics.enriched_mesh import EnrichedMesh 
import core.fenics.io_wrappers as iofe

from mesh import CookMembrane

comm = MPI.COMM_WORLD
comm_self = MPI.COMM_SELF
rank = comm.Get_rank()
num_ranks = comm.Get_size()

def solve_cook(meshfile, tangent_dataset):

    mesh = EnrichedMesh(meshfile)

    Uh = df.VectorFunctionSpace(mesh, "CG", 2)
    clampedBndFlag = 2 
    LoadBndFlag = 1 
    
    ty = 0.05
    traction = df.Constant((0.0,ty ))
    bcL = df.DirichletBC(Uh, df.Constant((0.0,0.0)), mesh.boundaries, clampedBndFlag)
    
    mapping = np.random.randint(0,len(tangent_dataset),mesh.num_cells())
        
    Chom = fmts.Chom_multiscale(tangent_dataset, mapping, degree = 0)
    
    # # Define variational problem
    uh = df.TrialFunction(Uh) 
    vh = df.TestFunction(Uh)
    a = df.inner(df.dot(Chom,symgrad_voigt(uh)), symgrad_voigt(vh))*mesh.dx
    b = df.inner(traction,vh)*mesh.ds(LoadBndFlag)
    
    # Compute solution
    uh = feut.solve_iterative(a, b, bcL, Uh)
    
    return uh, Chom


if __name__ == '__main__':
    
    # =========== argument input =================
    if(len(sys.argv)>1):
        Ny_split = int(sys.argv[1])
        caseType = sys.argv[2]
        seed = int(sys.argv[3])
    
    else:     
        Ny_split =  40 # 5, 10, 20, 40, 80
        caseType = 'reduced_per' # opt: reduced_per, dnn, full
        seed = 1
        
    
    # ========== dataset folders ================= 
    rootDataPath = open('../../rootDataPath.txt','r').readline()[:-1]
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
    
    sigma = lambda u: df.dot(Chom, symgrad_voigt(u))
    
    with df.XDMFFile(comm, folderMesh + "cook_%s_%d_vtk.xdmf"%(caseType,Ny_split)) as file:
        iofe.export_XDMF_displacement_sigma(uh, sigma, file)
    
    with df.XDMFFile(comm, folderMesh + "cook_%s_%d.xdmf"%(caseType,Ny_split)) as file:
        iofe.export_checkpoint_XDMF_displacement_sigma(uh, sigma, file)






