import sys, os
import dolfin as df 
import matplotlib.pyplot as plt
from ufl import nabla_div
sys.path.insert(0,'../../utils/')

from fenicsUtils import symgrad, symgrad_voigt
import numpy as np

import fenicsMultiscale as fmts
import myHDF5 as myhd
import meshUtils as meut
from timeit import default_timer as timer
import multiphenics as mp

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_self = MPI.COMM_SELF
rank = comm.Get_rank()
num_ranks = comm.Get_size()

rootDataPath = open('../../../rootDataPath.txt','r').readline()[:-1]

Ny_DNS = int(sys.argv[1])
problemType = sys.argv[2] if len(sys.argv)>3 else ''
caseType = sys.argv[3] if len(sys.argv)>3 else '' 

folder = rootDataPath + "/new_fe2/DNS/DNS_%d_new_bending/"%Ny_DNS

Lx = 2.0
Ly = 0.5

if(problemType == '_bending'):
    print("solving bending")
    ty = -0.1
    tx = 0.0
    
    tractionBnd = df.CompiledSubDomain('near(x[1], Ly) && on_boundary', Ly = Ly)
    clampedBnd = df.CompiledSubDomain('(near(x[0], Lx) || near(x[0], 0.0) ) && on_boundary', Lx = Lx)
    
    
elif(problemType == '_rightClamped'):
    print("solving rightClamped")
    ty = -0.01
    tx = 0.0
    
    tractionBnd = df.CompiledSubDomain('near(x[0], 0.0) && on_boundary')
    clampedBnd = df.CompiledSubDomain('near(x[0], Lx) && on_boundary', Lx = Lx)

    
elif(problemType == '_pulling'):
    print("solving pulling")
    ty = 0.0
    tx = 0.1
    
    clampedBnd = df.CompiledSubDomain('near(x[0], 0.0) && on_boundary')
    tractionBnd = df.CompiledSubDomain('near(x[0], Lx) && on_boundary', Lx = Lx)
        
else: ## standard shear on right clamped on left
    print("solving leftClamped")    
    ty = -0.01
    tx = 0.0    
    
    clampedBnd = df.CompiledSubDomain('near(x[0], 0.0) && on_boundary')
    tractionBnd = df.CompiledSubDomain('near(x[0], Lx) && on_boundary', Lx = Lx)

class myChom(df.UserExpression):
    def __init__(self, tangent, center,  **kwargs):
        self.tangent = tangent
        self.center = center
        
        super().__init__(**kwargs)
        
    def eval_cell(self, values, x, cell):
        dist = np.linalg.norm(self.center - x, axis = 1)
        values[:] = self.tangent[np.argmin(dist),:,:].flatten()
        
    def value_shape(self):
        return (3,3,)
    

# loading boundary reference mesh
# Ny = 96
# Nx = 4*Ny

# Create mesh and define function space
# mesh = df.RectangleMesh(comm,df.Point(0.0, 0.0), df.Point(Lx, Ly), Nx, Ny, "right/left")
# with df.XDMFFile(comm, folder + "multiscale/meshBarMacro_Multiscale_96.xdmf") as file:
#     file.write(mesh)
    
# Create mesh and define function space
mesh = df.Mesh(comm)
with df.XDMFFile(comm, folder + "multiscale/meshBarMacro_Multiscale.xdmf") as file:
    file.read(mesh)

Uh = df.VectorFunctionSpace(mesh, "CG", 2)

boundary_markers = df.MeshFunction("size_t", mesh, dim=1, value=0)
clampedBnd.mark(boundary_markers, 1)
tractionBnd.mark(boundary_markers, 2)

tangentName = folder + 'tangents/tangent_%s.hd5'%caseType
tangent = myhd.loadhd5(tangentName, 'tangent')
ids = myhd.loadhd5(tangentName, 'id')
center = myhd.loadhd5(tangentName, 'center')
# center = myhd.loadhd5(folder + 'ellipseData_RVEs.hd5', 'center') # temporary

# already sorted
sortIndex = np.argsort(ids)
tangent = tangent[sortIndex,:,:]
center = center[sortIndex,:] # temporary commented
ids = ids[sortIndex]

Chom = myChom(tangent, center, degree = 0)


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
uh = df.Function(Uh)
df.solve(a == b,uh, bcs = bcL, solver_parameters={"linear_solver": "superlu"})


with df.XDMFFile(comm, folder + "multiscale/barMacro_Multiscale_%s_vtk.xdmf"%(caseType)) as file:
    uh.rename('u','name')
    file.write(uh)

with df.XDMFFile(comm, folder + "multiscale/barMacro_Multiscale_%s.xdmf"%(caseType)) as file:
    file.write_checkpoint(uh,'u',0)
    
