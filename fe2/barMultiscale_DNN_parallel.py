#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Availabel in: https://github.com/felipefr/micmacsFenics.git
@author: Felipe Figueredo Rocha, f.rocha.felipe@gmail.com, felipe.figueredorocha@epfl.ch
   
"""

import sys, os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import dolfin as df 
import matplotlib.pyplot as plt
from ufl import nabla_div
sys.path.insert(0, '/home/felipefr/github/micmacsFenics/utils/')
sys.path.insert(0,'../utils/')

import multiscaleModels as mscm
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
    def __init__(self, microModels,  **kwargs):
        self.microModels = microModels
        
        super().__init__(**kwargs)
        
    def eval_cell(self, values, x, cell):
        values[:] = self.microModels[cell.index].getTangent().flatten()
        
    def value_shape(self):
        return (3,3,)

class MicroConstitutiveModelDNN(mscm.MicroConstitutiveModel):
    
    def __init__(self,nameMesh, param, model):
        self.nameMesh = nameMesh
        self.param = param
        self.model = model
        # it should be modified before computing tangent (if needed) 
        self.others = {'polyorder' : 1} 
        
        self.multiscaleModel = mscm.listMultiscaleModels[model]              
        self.ndim = 2
        self.nvoigt = int(self.ndim*(self.ndim + 1)/2)  
        self.Chom_ = np.zeros((self.nvoigt,self.nvoigt))
 
        self.getTangent = self.computeTangent # in the first run should compute     
            
    def readMesh(self):
        self.mesh = meut.EnrichedMesh(self.nameMesh,comm_self)
        self.lame = elut.getLameInclusions(*self.param, self.mesh)
        self.coord_min = np.min(self.mesh.coordinates(), axis = 0)
        self.coord_max = np.max(self.mesh.coordinates(), axis = 0)
        self.others['x0'] = self.coord_min[0]
        self.others['x1'] = self.coord_max[0] 
        self.others['y0'] = self.coord_min[1]
        self.others['y1'] = self.coord_max[1]
    
    def computeTangent(self):      
        
        self.readMesh()
        sigmaLaw = lambda u: self.lame[0]*nabla_div(u)*df.Identity(2) + 2*self.lame[1]*symgrad(u)
        
        dy = self.mesh.dx # specially for the case of enriched mesh, otherwise it does not work
        vol = df.assemble(df.Constant(1.0)*dy(0)) + df.assemble(df.Constant(1.0)*dy(1))
        
        y = df.SpatialCoordinate(self.mesh)
        Eps = df.Constant(((0.,0.),(0.,0.))) # just placeholder
        
        form = self.multiscaleModel(self.mesh, sigmaLaw, Eps, self.others)
        a,f,bcs,W = form()

        start = timer()        
        A = mp.block_assemble(a)
        if(len(bcs) > 0): 
            bcs.apply(A)
        
        solver = df.PETScLUSolver('superlu')
        sol = mp.BlockFunction(W)
             
        for i in range(self.nvoigt):
            start = timer()              
            self.others['uD'].vector().set_local(self.others['uD{0}_'.format(i)])
            
            Eps.assign(df.Constant(mscm.macro_strain(i)))    
            F = mp.block_assemble(f)
            if(len(bcs) > 0):
                bcs.apply(F)
        
            solver.solve(A,sol.block_vector(), F)    
            sol.block_vector().block_function().apply("to subfunctions")
            
            sig_mu = sigmaLaw(df.dot(Eps,y) + sol[0])
            sigma_hom =  sum([Integral(sig_mu, dy(i), (2,2)) for i in [0,1]])/vol
            
            self.Chom_[:,i] = sigma_hom.flatten()[[0,3,1]]
            
            end = timer()
            print('time in solving system', end - start) # Time in seconds
        
        self.getTangent = self.getTangent_ # from the second run onwards, just returns  
        
        return self.Chom_

    
# loading boundary reference mesh
nameMeshRefBnd = 'boundaryMesh.xdmf'
Mref = meut.EnrichedMesh(nameMeshRefBnd,comm_self)
Vref = df.VectorFunctionSpace(Mref,"CG", 1)

dxRef = df.Measure('dx', Mref) 


Lx = 2.0
Ly = 0.5
Nx = 10
Ny = 4
ty = -0.01

# Create mesh and define function space
mesh = df.RectangleMesh(comm,df.Point(0.0, 0.0), df.Point(Lx, Ly), Nx, Ny, "right/left")
with df.XDMFFile(comm, "meshBarMacro.xdmf") as file:
    file.write(mesh)
    
Uh = df.VectorFunctionSpace(mesh, "Lagrange", 1)

leftBnd = df.CompiledSubDomain('near(x[0], 0.0) && on_boundary')
rightBnd = df.CompiledSubDomain('near(x[0], Lx) && on_boundary', Lx = Lx)

boundary_markers = df.MeshFunction("size_t", mesh, dim=1, value=0)
leftBnd.mark(boundary_markers, 1)
rightBnd.mark(boundary_markers, 2)


dx = df.Measure('dx', domain=mesh)   

# defining the micro model

BCname = 'BCsPrediction.hd5'

contrast = 10.0
E2 = 1.0
nu = 0.3
param = [nu,E2*contrast,nu,E2]

microModelList = []


print(mesh.num_cells())
for cell in df.cells(mesh):
    i = cell.global_index()
    meshMicroName = './meshes/mesh_micro_{0}_full.xdmf'.format(i)
    microModelList.append( MicroConstitutiveModelDNN(meshMicroName, param, 'per') )
    microModelList[-1].others['uD'] = df.Function(Vref) 
    microModelList[-1].others['uD0_'] = myhd.loadhd5(BCname, 'u0')[i,:] 
    microModelList[-1].others['uD1_'] = myhd.loadhd5(BCname, 'u1')[i,:]
    microModelList[-1].others['uD2_'] = myhd.loadhd5(BCname, 'u2')[i,:]


Chom = myChom(microModelList, degree = 0)

# print(df.assemble(Chom[0,0]*dx))

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
df.solve(a == b,uh, bcs = bcL, solver_parameters={"linear_solver": "mumps"})


with df.XDMFFile(comm, "barMultiscale_per_full_vtk.xdmf") as file:
    uh.rename('u','name')
    file.write(uh)

with df.XDMFFile(comm, "barMultiscale_per_full.xdmf") as file:
    file.write_checkpoint(uh,'u',0)
    
