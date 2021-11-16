import sys, os
import dolfin as df 
import matplotlib.pyplot as plt
from ufl import nabla_div

import numpy as np
from timeit import default_timer as timer
from mpi4py import MPI

from deepBND.__init__ import *
from deepBND.core.fenics_tools.misc import Integral 
from deepBND.core.elasticity.fenics_utils import symgrad_voigt
from deepBND.core.fenics_tools.wrapper_solvers import solver_iterative
from deepBND.core.multiscale.misc import Chom_multiscale, Chom_precomputed_dist
import deepBND.core.data_manipulation.wrapper_h5py as myhd
from deepBND.core.fenics_tools.enriched_mesh import EnrichedMesh 
     
comm = MPI.COMM_WORLD
comm_self = MPI.COMM_SELF
rank = comm.Get_rank()
num_ranks = comm.Get_size()

def solve_barMultiscale(meshfile, tangentName,  param):
    
    Lx, Ly, tx, ty = param 
    
    clampedBnd = df.CompiledSubDomain('near(x[0], 0.0) && on_boundary')
    tractionBnd = df.CompiledSubDomain('near(x[0], Lx) && on_boundary', Lx = Lx)
    
    # Create mesh and define function space
    mesh = df.Mesh(comm)
    with df.XDMFFile(comm, meshfile) as file:
        file.read(mesh)
    
    Uh = df.VectorFunctionSpace(mesh, "CG", 2)
    
    boundary_markers = df.MeshFunction("size_t", mesh, dim=1, value=0)
    clampedBnd.mark(boundary_markers, 1)
    tractionBnd.mark(boundary_markers, 2)
    
    tangent = myhd.loadhd5(tangentName, 'tangent')
    ids = myhd.loadhd5(tangentName, 'id')
    center = myhd.loadhd5(tangentName, 'center')
    # center = myhd.loadhd5(folder + 'ellipseData_RVEs.hd5', 'center') # temporary
    
    # already sorted
    sortIndex = np.argsort(ids)
    tangent = tangent[sortIndex,:,:]
    center = center[sortIndex,:] # temporary commented
    ids = ids[sortIndex]

    Chom = Chom_precomputed_dist(tangent, center, mesh, degree = 0)    
    
    # Define boundary condition
    bcL = df.DirichletBC(Uh, df.Constant((0.0,0.0)), boundary_markers, 1) # leftBnd instead is possible
    
    ds = df.Measure('ds', domain=mesh, subdomain_data=boundary_markers)
    dx = df.Measure('dx', domain=mesh)
    traction = df.Constant((0.0,ty ))
    
    # # Define variational problem
    uh = df.TrialFunction(Uh) 
    vh = df.TestFunction(Uh)
    a = df.inner(df.dot(Chom,symgrad_voigt(uh)), symgrad_voigt(vh))*dx
    b = df.inner(traction,vh)*ds(2)
    
    # # Compute solution
    
    uh = solver_iterative(a, b, bcL, Uh)

    return uh, Chom
    

if __name__ == '__main__':
    
    # =========== argument input =================
    if(len(sys.argv)>1):
        Ny_DNS = int(sys.argv[1])
        caseType = sys.argv[2]
        createMesh = int(sys.argv[3])
    
    else:     
        Ny_DNS =  24 # 24, 72
        caseType = 'reduced_per' # opt: reduced_per, dnn_big, fulls
        createMesh = 0
        
    folder = rootDataPath + "/deepBND/DNS/DNS_%d_new/"%Ny_DNS
    meshfile = folder + "multiscale/meshBarMacro_Multiscale.xdmf"
    tangentName = folder + 'tangents/tangent_%s.hd5'%caseType
    
    if(createMesh == 1):
        # loading boundary reference mesh
        Ny = 96
        Nx = 4*Ny
        
        # Create mesh and define function space
        mesh = df.RectangleMesh(comm,df.Point(0.0, 0.0), df.Point(Lx, Ly), Nx, Ny, "right/left")
        with df.XDMFFile(comm, meshfile) as file:
            file.write(mesh)
        
    Lx = 2.0
    Ly = 0.5
    
    ty = -0.01
    tx = 0.0    
    
    param = [Lx, Ly, tx, ty]
    
    uh, Chom = solve_barMultiscale(meshfile, tangentName, param)

    sigma = lambda u: df.dot(Chom, symgrad_voigt(u))
    
    mesh = uh.function_space().mesh()
    dx = df.Measure('dx', mesh)
    
    print(Integral(uh, dx, shape=(2,)))
    print(Integral(sigma(uh), dx, shape=(3,)))

    with df.XDMFFile(comm, folder + "multiscale/barMacro_Multiscale_%s_vtk.xdmf"%(caseType)) as file:
        uh.rename('u','name')
        file.write(uh)
    
    with df.XDMFFile(comm, folder + "multiscale/barMacro_Multiscale_%s.xdmf"%(caseType)) as file:
        file.write_checkpoint(uh,'u',0)
