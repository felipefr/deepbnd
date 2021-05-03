import sys, os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import dolfin as df 
import matplotlib.pyplot as plt
from ufl import nabla_div
sys.path.insert(0,'../../utils/')

from fenicsUtils import symgrad, symgrad_voigt, Integral
import numpy as np

import fenicsMultiscale as fmts
import myHDF5 as myhd
import meshUtils as meut
import elasticity_utils as elut
import symmetryLib as symlpy
from timeit import default_timer as timer
import multiphenics as mp

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_self = MPI.COMM_SELF
rank = comm.Get_rank()
num_ranks = comm.Get_size()

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
    
caseType = 'dnn'
volFrac = ''
jump = ''

Ny_DNS = 72
folder = "./DNS_{0}/".format(Ny_DNS)

# loading boundary reference mesh
Lx = 2.0
Ly = 0.5
Ny = 48
Nx = 4*Ny
ty = -0.01


# Create mesh and define function space
# mesh = df.RectangleMesh(comm,df.Point(0.0, 0.0), df.Point(Lx, Ly), Nx, Ny, "right/left")
# with df.XDMFFile(comm, folder + "multiscale{1}/meshBarMacro_Multiscale_ny{0}.xdmf".format(Ny,jump)) as file:
#     file.write(mesh)
    
# Create mesh and define function space
mesh = df.Mesh(comm)
with df.XDMFFile(comm, folder + "multiscale{1}/meshBarMacro_Multiscale_ny{0}.xdmf".format(Ny,jump)) as file:
    file.read(mesh)


    
Uh = df.VectorFunctionSpace(mesh, "CG", 2)

leftBnd = df.CompiledSubDomain('near(x[0], 0.0) && on_boundary')
rightBnd = df.CompiledSubDomain('near(x[0], Lx) && on_boundary', Lx = Lx)

boundary_markers = df.MeshFunction("size_t", mesh, dim=1, value=0)
leftBnd.mark(boundary_markers, 1)
rightBnd.mark(boundary_markers, 2)

tangentName = folder + 'tangents{0}/tangent_{1}.hd5'.format(volFrac,caseType)
tangent = myhd.loadhd5(tangentName, 'tangent')
ids = myhd.loadhd5(tangentName, 'id')
center = myhd.loadhd5(tangentName, 'center')
# center = myhd.loadhd5(folder + 'ellipseData_RVEs.hd5', 'center') # temporary

# already sorted
sortIndex = np.argsort(ids)
tangent = tangent[sortIndex,:,:]
center = center[sortIndex,:] # temporary commented
ids = ids[sortIndex]

if(jump == "_jump"):
    print("jumping some cells")
    ids = ids[::6]
    center = center[::6]
    tangent = tangent[::6]

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


with df.XDMFFile(comm, folder + "multiscale{3}/barMacro_Multiscale{0}_{1}_ny{2}_vtk.xdmf".format(volFrac,caseType,Ny,jump)) as file:
    uh.rename('u','name')
    file.write(uh)

with df.XDMFFile(comm, folder + "multiscale{3}/barMacro_Multiscale{0}_{1}_ny{2}.xdmf".format(volFrac,caseType,Ny,jump)) as file:
    file.write_checkpoint(uh,'u',0)
    
