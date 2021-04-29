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
    def __init__(self, tangent,  **kwargs):
        self.tangent = tangent
        
        super().__init__(**kwargs)
        
    def eval_cell(self, values, x, cell):
        values[:] = self.tangent[cell.index,:,:].flatten()
        
    def value_shape(self):
        return (3,3,)
    
caseType = 'reduced_dnn_medium_40'
# loading boundary reference mesh
Lx = 2.0
Ly = 0.5
Ny = 35
Nx = 4*Ny
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

tangentName = '../../tangents/tangent_{0}.hd5'.format(caseType)
tangent = myhd.loadhd5(tangentName, 'tangent')
ids = myhd.loadhd5(tangentName, 'id')
sortIndex = np.argsort(ids)
tangent = tangent[sortIndex,:,:]
ids = ids[sortIndex]
print(ids)

Chom = myChom(tangent, degree = 0)


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
df.solve(a == b,uh, bcs = bcL, solver_parameters={"linear_solver": "superlu_dist"})
# df.solve(a == b,uh, bcs = bcL, solver_parameters={"linear_solver": "cg", "preconditioner" : 'hypre_parasails' })


with df.XDMFFile(comm, "barMacro_{0}_vtk_test.xdmf".format(caseType)) as file:
    uh.rename('u','name')
    file.write(uh)

with df.XDMFFile(comm, "barMacro_{0}_test.xdmf".format(caseType)) as file:
    file.write_checkpoint(uh,'u',0)
    
