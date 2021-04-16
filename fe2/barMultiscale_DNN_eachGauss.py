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
from tensorflow_for_training import *
import tensorflow as tf
import symmetryLib as symlpy
from timeit import default_timer as timer
import multiphenics as mp

permY = np.array([2,0,3,1,12,10,8,4,13,5,14,6,15,11,9,7,30,28,26,24,22,16,31,17,32,18,33,19,34,20,35,29,27,25,23,21])

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

class myChom(df.UserExpression):
    def __init__(self, microModels,  **kwargs):
        self.microModels = microModels
        
        super().__init__(**kwargs)
        
    def eval_cell(self, values, x, cell):    
        values[:] = self.microModels[cell.index].getTangent().flatten()
        
    def value_shape(self):
        return (3,3,)
    
    
class MicroConstitutiveModelDNN(mscm.MicroConstitutiveModel):
    
    def computeTangent(self):
        
        dy = df.Measure('dx',self.mesh)
        vol = df.assemble(df.Constant(1.0)*dy)
        y = df.SpatialCoordinate(self.mesh)
        Eps = df.Constant(((0.,0.),(0.,0.))) # just placeholder
        
        form = self.multiscaleModel(self.mesh, self.sigmaLaw, Eps, self.others)
        a,f,bcs,W = form()

        start = timer()        
        A = mp.block_assemble(a)
        if(len(bcs) > 0): 
            bcs.apply(A)
        
        solver = df.LUSolver(A) # decompose just once
        sol = mp.BlockFunction(W)
             
        for i in range(self.nvoigt):
            
            self.others['uD'].vector().set_local(self.others['uD{0}_'.format(i)])
            
            Eps.assign(df.Constant(mscm.macro_strain(i)))    
            F = mp.block_assemble(f)
            if(len(bcs) > 0):
                bcs.apply(F)
        
            solver.solve(sol.block_vector(), F)    
            sol.block_vector().block_function().apply("to subfunctions")
            
            sig_mu = self.sigmaLaw(df.dot(Eps,y) + sol[0])
            sigma_hom =  Integral(sig_mu, dy, (2,2))/vol

            self.Chom_[:,i] = sigma_hom.flatten()[[0,3,1]]
            
        end = timer()
        print('time in solving system', end - start) # Time in seconds
        
        print(self.Chom_)
        
        self.getTangent = self.getTangent_ # from the second run onwards, just returns  
        
        return self.Chom_
               
    
    
# loading boundary reference mesh
nameMeshRefBnd = 'boundaryMesh.xdmf'
Mref = meut.EnrichedMesh(nameMeshRefBnd)
Vref = df.VectorFunctionSpace(Mref,"CG", 1)

dxRef = df.Measure('dx', Mref) 
    
# loading the DNN model
folder = './models/dataset_axial3/'
ellipseData = myhd.loadhd5(folder +  'ellipseData.h5', 'ellipseData')

Nrb = 140
archId = 1
nX = 36
folderBasisShear = './models/dataset_shear1/'
folderBasisAxial = './models/dataset_axial1/'
nameWbasisShear = folderBasisShear +  'Wbasis.h5'
nameWbasisAxial = folderBasisAxial +  'Wbasis.h5'
nameScaleXY_shear = folderBasisShear +  'scaler.txt'
nameScaleXY_axial = folderBasisAxial +  'scaler.txt'

net = {'Neurons': 3*[300], 'activations': 3*['swish'] + ['linear'], 'lr': 5.0e-4, 'decay' : 0.1, 'drps' : [0.0] + 3*[0.005] + [0.0], 'reg' : 1.0e-8}

net['nY'] = Nrb
net['nX'] = nX
net['file_weights_shear'] = folderBasisShear + 'models/weights_ny{0}_arch{1}.hdf5'.format(Nrb,archId)
net['file_weights_axial'] = folderBasisAxial + 'models/weights_ny{0}_arch{1}.hdf5'.format(Nrb,archId)

scalerX_shear, scalerY_shear  = importScale(nameScaleXY_shear, nX, Nrb)
scalerX_axial, scalerY_axial  = importScale(nameScaleXY_axial, nX, Nrb)

Wbasis_shear = myhd.loadhd5(nameWbasisShear, 'Wbasis')
Wbasis_axial = myhd.loadhd5(nameWbasisAxial, 'Wbasis')

X_shear_s = scalerX_shear.transform(ellipseData[:,:,2])
X_axial_s = scalerX_axial.transform(ellipseData[:,:,2])
X_axialY_s = scalerX_axial.transform(ellipseData[:,permY,2])

modelShear = generalModel_dropReg(nX, Nrb, net)   
modelAxial = generalModel_dropReg(nX, Nrb, net)   

modelShear.load_weights(net['file_weights_shear'])
modelAxial.load_weights(net['file_weights_axial'])

Y_p_shear = scalerY_shear.inverse_transform(modelShear.predict(X_shear_s))
Y_p_axial = scalerY_axial.inverse_transform(modelAxial.predict(X_axial_s))
Y_p_axialY = scalerY_axial.inverse_transform(modelAxial.predict(X_axialY_s)) # changed

S_p_shear = Y_p_shear @ Wbasis_shear[:Nrb,:]
S_p_axial = Y_p_axial @ Wbasis_axial[:Nrb,:]
piola_mat = syml.PiolaTransform_matricial('mHalfPi', Vref)
S_p_axialY = Y_p_axialY @ Wbasis_axial[:Nrb,:] @ piola_mat.T #changed

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
mesh = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(Lx, Ly), Nx, Ny, "right/left")
Uh = df.VectorFunctionSpace(mesh, "Lagrange", 1)

leftBnd = df.CompiledSubDomain('near(x[0], 0.0) && on_boundary')
rightBnd = df.CompiledSubDomain('near(x[0], Lx) && on_boundary', Lx = Lx)

boundary_markers = df.MeshFunction("size_t", mesh, dim=1, value=0)
leftBnd.mark(boundary_markers, 1)
rightBnd.mark(boundary_markers, 2)

# defining the micro model
i = 0
nCells = mesh.num_cells()

meshMicro = get_mesh(ellipseData[i])

contrast = 10.0
E2 = 1.0
nu = 0.3

lame = elut.getLameInclusions(nu,E2*contrast,nu,E2,meshMicro)

microModel = MicroConstitutiveModelDNN(meshMicro, [lame[0],lame[1]], 'lin')

microModel.others['uD'] = df.Function(Vref) 
microModel.others['uD0_'] = S_p_axial[i,:] 
microModel.others['uD1_'] = S_p_axialY[i,:]
microModel.others['uD2_'] = S_p_shear[i,:] 

Chom = myChom(nCells*[microModel], degree = 0)

# Define boundary condition
bcL = df.DirichletBC(Uh, df.Constant((0.0,0.0)), boundary_markers, 1) # leftBnd instead is possible

ds = df.Measure('ds', domain=mesh, subdomain_data=boundary_markers)
dx = df.Measure('dx', domain=mesh)
traction = df.Constant((0.0,ty ))


# Define variational problem
uh = df.TrialFunction(Uh) 
vh = df.TestFunction(Uh)
a = df.inner(df.dot(Chom,symgrad_voigt(uh)), symgrad_voigt(vh))*dx
b = df.inner(traction,vh)*ds(2)

# Compute solution
uh = df.Function(Uh)
df.solve(a == b, uh, bcL)

# Save solution in VTK format
fileResults = df.XDMFFile("barMultiscale.xdmf")
fileResults.write(uh)

print(uh(df.Point(2.0,0.0)))

# 'DNN'
# [[ 1.66684800e+00  4.59212534e-01  4.70612154e-05]
#  [ 4.59239457e-01  1.66189055e+00 -8.86338031e-04]
#  [ 2.90270943e-04 -6.57664118e-04  5.40716823e-01]]
# [-0.11742746 -0.65979363]

# "per"
# [[ 1.66280941e+00  4.59841619e-01  1.11955710e-06]
#  [ 4.59841619e-01  1.66394286e+00 -1.41096685e-06]
#  [ 1.11955751e-06 -1.41096684e-06  5.40675732e-01]]
# [-0.11772728 -0.66135198]

# lin
# [[1.66970065e+00 4.61283616e-01 2.20028672e-05]
#  [4.61283616e-01 1.67009647e+00 1.93489513e-05]
#  [2.20028668e-05 1.93489512e-05 5.60790992e-01]]
# [-0.11698271 -0.65603435]

# MR 
# [[ 1.62073535e+00  4.91553331e-01 -1.98170109e-05]
#  [ 4.91553331e-01  1.62086580e+00 -2.28521317e-05]
#  [-1.98170114e-05 -2.28521318e-05  5.35827062e-01]]
# [-0.12271125 -0.68772474]
