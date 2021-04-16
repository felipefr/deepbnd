#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Availabel in: https://github.com/felipefr/micmacsFenics.git
@author: Felipe Figueredo Rocha, f.rocha.felipe@gmail.com, felipe.figueredorocha@epfl.ch
   
"""

import sys, os
from dolfin import *
import matplotlib.pyplot as plt
from ufl import nabla_div
sys.path.insert(0, '/home/felipefr/github/micmacsFenics/utils/')

import multiscaleModels as mscm
from fenicsUtils import symgrad, symgrad_voigt
import numpy as np

sys.path.insert(0,'../utils/')
import fenicsMultiscale as fmts
import myHDF5 as myhd
import meshUtils as meut
import elasticity_utils as elut

def get_mesh(ellipseData):
    maxOffset = 2
    
    H = 1.0 # size of each square
    NxL = NyL = 2
    NL = NxL*NyL
    x0L = y0L = -H 
    LxL = LyL = 2*H
    lcar = (2/30)*H
    Nx = (NxL+2*maxOffset)
    Ny = (NyL+2*maxOffset)
    Lxt = Nx*H
    Lyt = Ny*H
    NpLxt = int(Lxt/lcar) + 1
    NpLxL = int(LxL/lcar) + 1
    print("NpLxL=", NpLxL) 
    x0 = -Lxt/2.0
    y0 = -Lyt/2.0
    r0 = 0.2*H
    r1 = 0.4*H
    Vfrac = 0.282743
    rm = H*np.sqrt(Vfrac/np.pi)
    
    meshGMSH = meut.ellipseMesh2(ellipseData[:4,:], x0L, y0L, LxL, LyL, lcar)
    meshGMSH.setTransfiniteBoundary(NpLxL)
        
    meshGMSH.setNameMesh("mesh_micro.xml")
    mesh = meshGMSH.getEnrichedMesh()

            
    return mesh

class myChom(UserExpression):
    def __init__(self, microModels,  **kwargs):
        self.microModels = microModels
        
        super().__init__(**kwargs)
        
    def eval_cell(self, values, x, cell):    
        values[:] = self.microModels[cell.index].getTangent().flatten()
        
    def value_shape(self):
        return (3,3,)


r0 = 0.3
r1 = 0.5
Lx = 2.0
Ly = 0.5
Nx = 10
Ny = 3

lamb_matrix = 1.0
mu_matrix = 0.5
NxMicro = NyMicro = 100
LxMicro = LyMicro = 1.0
contrast = 10.0

ty = -0.01

# Create mesh and define function space
mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), Nx, Ny, "right/left")
Uh = VectorFunctionSpace(mesh, "Lagrange", 1)

leftBnd = CompiledSubDomain('near(x[0], 0.0) && on_boundary')
rightBnd = CompiledSubDomain('near(x[0], Lx) && on_boundary', Lx = Lx)

boundary_markers = MeshFunction("size_t", mesh, dim=1, value=0)
leftBnd.mark(boundary_markers, 1)
rightBnd.mark(boundary_markers, 2)

# defining the micro model
nCells = mesh.num_cells()
folder = './models/dataset_axial3/'
ellipseData = myhd.loadhd5(folder +  'ellipseData.h5', 'ellipseData')[0,:,:] 
meshMicro = get_mesh(ellipseData)

contrast = 10.0
E2 = 1.0
nu = 0.3

lame = elut.getLameInclusions(nu,E2*contrast,nu,E2,meshMicro)

microModel = mscm.MicroConstitutiveModel(meshMicro, [lame[0],lame[1]], 'per') 

Chom = myChom(nCells*[microModel], degree = 0)

# Define boundary condition
bcL = DirichletBC(Uh, Constant((0.0,0.0)), boundary_markers, 1) # leftBnd instead is possible

ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
traction = Constant((0.0,ty ))


# Define variational problem
uh = TrialFunction(Uh) 
vh = TestFunction(Uh)
a = inner(dot(Chom,symgrad_voigt(uh)), symgrad_voigt(vh))*dx
b = inner(traction,vh)*ds(2)

# Compute solution
uh = Function(Uh)
solve(a == b, uh, bcL)

# Save solution in VTK format
fileResults = XDMFFile("barMultiscale.xdmf")
fileResults.write(uh)

print(uh(Point(2.0,0.0)))



