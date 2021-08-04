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
import fenicsUtils as feut
from cook_mesh import *
import ioFenicsWrappers as iofe
from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_self = MPI.COMM_SELF
rank = comm.Get_rank()
num_ranks = comm.Get_size()


Ny_split = int(sys.argv[1])
caseType = sys.argv[2]
seed = int(sys.argv[3])

# # Ny_split =  50

# caseType = 'reduced_per'
# Ny_split =  50
# Ny_split = 100

rootDataPath = open('../../../rootDataPath.txt','r').readline()[:-1]
folder = rootDataPath + "/new_fe2/DNS/DNS_72_2/"


folderMesh = rootDataPath + '/new_fe2/cook/meshes_{0}/'.format(seed)
# folderMesh = './'

tangentName = folder + 'tangents/tangent_%s.hd5'%caseType
tangent = myhd.loadhd5(tangentName, 'tangent')
ids = myhd.loadhd5(tangentName, 'id')
center = myhd.loadhd5(tangentName, 'center')

lcar = 44.0/Ny_split  

# gmshMesh = CookMembrane(lcar = lcar)
# gmshMesh.write(savefile = folderMesh + 'mesh_%d.xdmf'%Ny_split, opt = 'fenics')
mesh = meut.EnrichedMesh(folderMesh + 'mesh_%d.xdmf'%Ny_split)

Uh = df.VectorFunctionSpace(mesh, "CG", 2)
clampedBndFlag = 2 
LoadBndFlag = 1 

# ty = 0.05797253785428244
ty = 0.05
traction = df.Constant((0.0,ty ))
bcL = df.DirichletBC(Uh, df.Constant((0.0,0.0)), mesh.boundaries, clampedBndFlag)

class myChom(df.UserExpression):
    def __init__(self, tangent, mapping,  **kwargs):
        self.tangent = tangent
        self.map = mapping
        
        super().__init__(**kwargs)
        
    def eval_cell(self, values, x, cell):
        values[:] = self.tangent[self.map[cell.index],:,:].flatten()
        
    def value_shape(self):
        return (3,3,)
    

np.random.seed(seed)
mapping = np.random.randint(0,len(tangent),mesh.num_cells())
    
Chom = myChom(tangent, mapping, degree = 0)

# # Define variational problem
uh = df.TrialFunction(Uh) 
vh = df.TestFunction(Uh)
a = df.inner(df.dot(Chom,symgrad_voigt(uh)), symgrad_voigt(vh))*mesh.dx
b = df.inner(traction,vh)*mesh.ds(LoadBndFlag)

# # Compute solution

uh = feut.solve_iterative(a, b, bcL, Uh)



with df.XDMFFile(comm, folderMesh + "cook_%s_%d_vtk.xdmf"%(caseType,Ny_split)) as file:

    file.parameters["flush_output"] = True
    file.parameters["functions_share_mesh"] = True
    
    uh.rename('u','displacements at nodes')
    file.write(uh, 0.)
    
    sigma = lambda u: df.dot(Chom, symgrad_voigt(u))
    voigt2ten = lambda a: df.as_tensor(((a[0],a[2]),(a[2],a[1])))
    Vsig = df.FunctionSpace(mesh, "DG", 0)
    von_Mises = df.Function(Vsig, name="vonMises")
    sig_xx = df.Function(Vsig, name="sig_xx")
    sig_yy = df.Function(Vsig, name="sig_yy")
    sig_xy = df.Function(Vsig, name="sig_xy")
    
    sig = voigt2ten(sigma(uh))
    s = sig - (1./3)*df.tr(sig)*df.Identity(2)
    von_Mises_ = df.sqrt((3./2)*df.inner(s, s)) 
    von_Mises.assign(iofe.local_project(von_Mises_, Vsig))
    sig_xx.assign(iofe.local_project(sig[0,0], Vsig))
    sig_yy.assign(iofe.local_project(sig[1,1], Vsig))
    sig_xy.assign(iofe.local_project(sig[0,1], Vsig))
    
    file.write(von_Mises,0.)
    file.write(sig_xx,0.)
    file.write(sig_yy,0.)
    file.write(sig_xy,0.)


with df.XDMFFile(comm, folderMesh + "cook_%s_%d.xdmf"%(caseType,Ny_split)) as file:

    file.parameters["flush_output"] = True
    file.parameters["functions_share_mesh"] = True
    
    
    sigma = lambda u: df.dot(Chom, symgrad_voigt(u))
    voigt2ten = lambda a: df.as_tensor(((a[0],a[2]),(a[2],a[1])))
    Vsig = df.FunctionSpace(mesh, "DG", 0)
    von_Mises = df.Function(Vsig, name="vonMises")
    sig_xx = df.Function(Vsig, name="sig_xx")
    sig_yy = df.Function(Vsig, name="sig_yy")
    sig_xy = df.Function(Vsig, name="sig_xy")
    
    sig = voigt2ten(sigma(uh))
    s = sig - (1./3)*df.tr(sig)*df.Identity(2)
    von_Mises_ = df.sqrt((3./2)*df.inner(s, s)) 
    von_Mises.assign(iofe.local_project(von_Mises_, Vsig))
    sig_xx.assign(iofe.local_project(sig[0,0], Vsig))
    sig_yy.assign(iofe.local_project(sig[1,1], Vsig))
    sig_xy.assign(iofe.local_project(sig[0,1], Vsig))

    file.write_checkpoint(von_Mises,'vonMises', 0, df.XDMFFile.Encoding.HDF5)
    file.write_checkpoint(sig_xx,'sig_xx', 0, df.XDMFFile.Encoding.HDF5, True)
    file.write_checkpoint(sig_yy,'sig_yy', 0, df.XDMFFile.Encoding.HDF5, True)
    file.write_checkpoint(sig_xy,'sig_xy', 0, df.XDMFFile.Encoding.HDF5, True)
    
    file.write_checkpoint(uh,'u', 0, df.XDMFFile.Encoding.HDF5, True)
    





