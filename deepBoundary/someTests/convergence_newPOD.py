import sys, os
from numpy import isclose
from dolfin import *
# import matplotlib.pyplot as plt
from multiphenics import *
sys.path.insert(0, '../../utils/')

import fenicsWrapperElasticity as fela
import matplotlib.pyplot as plt
import numpy as np
import generatorMultiscale as gmts
import wrapperPygmsh as gmsh
import generationInclusions as geni
import myCoeffClass as coef
import fenicsMultiscale as fmts
import elasticity_utils as elut
import fenicsWrapperElasticity as fela
import multiphenicsMultiscale as mpms
import fenicsUtils as feut

from timeit import default_timer as timer

from mpi4py import MPI as pyMPI
import pickle

comm = MPI.comm_world

normL2 = lambda x,dx : np.sqrt(assemble(inner(x,x)*dx))

parameters["form_compiler"]["representation"] = "uflacs"

def local_project(v, V, dxm):
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_)*dxm
    b_proj = inner(v, v_)*dxm
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    
    u = Function(V)
    solver.solve_local_rhs(u)
    return u

folder = "../data1/"
radFile = folder + "RVE_POD_{0}.{1}"

BC = 'MR'
BCtest = 'MR'

offset = 2
Lx = Ly = 1.0
ifPeriodic = False 
NxL = NyL = 2
NL = NxL*NyL
x0L = y0L = offset*Lx/(NxL+2*offset)
LxL = LyL = offset*Lx/(NxL+2*offset)
r0 = 0.2*LxL/NxL
r1 = 0.4*LxL/NxL
times = 1
lcar = 0.1*LxL/NxL
NpLx = int(Lx/lcar) + 1
NpLxL = int(LxL/lcar) + 1
Vfrac = 0.282743

contrast = 10.0
E2 = 1.0
E1 = contrast*E2 # inclusions
nu1 = 0.3
nu2 = 0.3

ns = 400

Nbasis = np.array([4,8,12,18,25,50,75,100,125,150,156])
Nmax = np.max(Nbasis) + 1

Ntest = 1
NNbasis = len(Nbasis) 

EpsFluc = np.loadtxt(folder + 'Eps{0}.txt'.format(BCtest))
StressList = np.loadtxt(folder + 'sigmaList.txt')

S = []
for i in range(ns):    
    print('loading simulation ' + str(i) )
    mesh = fela.EnrichedMesh(radFile.format('reduced_' + str(i),'xml'))
    V = VectorFunctionSpace(mesh,"CG", 1)
    utemp = Function(V)
        
    with HDF5File(comm, radFile.format('solution_red_' + str(i),'h5'), 'r') as f:
        f.read(utemp, 'basic')
        S.append(utemp)

# ns = 2
C = []
for i in range(2):
    C.append(np.zeros((ns,ns)))

# for i in range(ns):
#     print('mounting line of Correlation ' + str(i) )
#     V = S[i].function_space()
#     mesh = V.mesh()
    
#     mesh = fela.EnrichedMesh(radFile.format('reduced_' + str(i),'xml'))
#     V = VectorFunctionSpace(mesh,"CG", 1)
    
#     for k in range(6):
#         C[k][i,i] = assemble(inner(grad(S[i]),grad(S[i]))*mesh.dx, form_compiler_parameters = {'quadrature_degree': k+1})
    
#     for j in range(i+1,ns):
#         ISj = interpolate(S[j],V)
#         for k in range(6):
#             C[k][i,j] = assemble(inner(grad(S[i]),grad(ISj))*mesh.dx, form_compiler_parameters = {'quadrature_degree': k+1})  
#             C[k][j,i] = C[k][i,j]


# for i in range(ns):
#     print('mounting line of Correlation ' + str(i) )

#     mesh = fela.EnrichedMesh(radFile.format('reduced_' + str(i),'xml'))
#     mesh = refine(mesh)
#     V = VectorFunctionSpace(mesh,"CG", 1)
    
#     ISi = interpolate(S[i],V)
    
#     for j in range(i,ns):
#         ISj = interpolate(S[j],V)
#         for k in range(6):
#             C[k][i,j] = assemble(inner(grad(ISi),grad(ISj))*dx, form_compiler_parameters = {'quadrature_degree': k+1})  
#             C[k][j,i] = C[k][i,j]


for i in range(ns):
    print('mounting line of Correlation ' + str(i) )

    mesh = fela.EnrichedMesh(radFile.format('reduced_' + str(i),'xml'))

    for k in range(2):
        # quad_element = FiniteElement(family = "Quadrature", cell = mesh.ufl_cell(), degree = 1, quad_scheme="default")
        
        # metadata = {"quadrature_degree": 2, "quadrature_scheme": "default"}
        # dxm = mesh.dx(metadata=metadata)
        
        # Q = FunctionSpace(mesh, quad_element, 0)
            
        # ISi = local_project(div(S[i]), Q, dxm)
        
        for j in range(i,ns):
            # ISj = project(grad(S[j]),Q)
            
            C[k][i,j] = assemble(inner(grad(S[i]), grad(S[j]))*dx, form_compiler_parameters = {'quadrature_degree': k+1})  
            C[k][j,i] = C[k][i,j]


# for i in range(ns):
#     print('mounting line of Correlation ' + str(i) )
    
#     mesh = S[i].function_space().mesh()
#     mesh = refine(refine(mesh)) # one refinement already computed
    
#     V = VectorFunctionSpace(mesh,"CG", 1)

#     ISi = interpolate(S[i],V)

#     for j in range(i,ns):
#         ISj = interpolate(S[j],V)
#         C[0][i,j] = assemble(inner(grad(ISi),grad(ISj))*dx)  
#         C[0][j,i] = C[0][i,j]


for k in range(1):
    np.savetxt('C_quad_new_' + str(k+1) + '.txt', C[k])

