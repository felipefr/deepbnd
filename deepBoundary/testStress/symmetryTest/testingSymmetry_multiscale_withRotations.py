import sys, os
from numpy import isclose
from dolfin import *
from multiphenics import *
sys.path.insert(0, '../../../utils/')

import fenicsWrapperElasticity as fela
import matplotlib.pyplot as plt
import numpy as np
import meshUtils as meut
import generationInclusions as geni
import fenicsMultiscale as fmts
import elasticity_utils as elut
import fenicsWrapperElasticity as fela
import multiphenicsMultiscale as mpms
import ioFenicsWrappers as iofe
import fenicsUtils as feut
import myHDF5 as myhd
import matplotlib.pyplot as plt
import copy
from timeit import default_timer as timer

from skopt.space import Space
from skopt.sampler import Lhs

import symmetryLib as syml
from ufl import nabla_div

import symmetryLib as syml

np.random.seed(3)
r0 = 0.1
r1 = 0.4
NxL = NyL = 1
x0 = y0 = -1.0
Lyt = Lxt = 2.0
maxOffset = 2

kindTransformation = 'rotation'
codeTransformation = 2

nxy = 100
Mref = RectangleMesh(Point(x0, y0), Point(x0+Lxt, y0+Lyt), nxy, nxy, "right/left")
Vref = VectorFunctionSpace(Mref,"CG", 2)

x = SpatialCoordinate(Mref)

if(kindTransformation == 'reflection'):
    s0, s1 = [ (-1.0,1.0), (1.0,-1.0), (-1.0,-1.0) ][codeTransformation]
    B = np.array([[s0,0.0],[0.0,s1]])
    T = [syml.T_horiz, syml.T_vert, syml.T_diag][codeTransformation]
        
elif(kindTransformation == 'rotation'):
    theta = [ 0.5*np.pi, np.pi, -0.5*np.pi][codeTransformation]
    s = np.sin(theta)
    c = np.cos(theta)
    B = np.array([[c,-s],[s,c]])
    T = [syml.T_halfpi, syml.T_pi, syml.T_mhalfpi][codeTransformation]
    
Finv = feut.affineTransformationExpession(np.zeros(2), B.T, Mref)
N = (2*maxOffset + NxL)*(2*maxOffset + NyL)
perm = syml.perm_with_order1(T, N, NxL)  

ellipseData, PermTotal, PermBox = geni.circularRegular2Regions(r0, r1, NxL, NyL, Lxt, Lyt, maxOffset, ordered = False, x0 = x0, y0 = y0)
ellipseData = ellipseData[PermTotal]

ellipseDataT = copy.deepcopy(ellipseData)
ellipseDataT[:,2] = ellipseDataT[perm,2] 

fac = Expression('1.0', degree = 2) # ground substance
facT = Expression('1.0', degree = 2) # ground substance
radiusThreshold = 0.01

for xi, yi, ri in ellipseData[:,0:3]:
    fac = fac + Expression('exp(-a*( (x[0] - x0)*(x[0] - x0) + (x[1] - y0)*(x[1] - y0) ) )', 
                           a = - np.log(radiusThreshold)/ri**2, x0 = xi, y0 = yi, degree = 2)

for xi, yi, ri in ellipseDataT[:,0:3]:
    facT = facT + Expression('exp(-a*( (x[0] - x0)*(x[0] - x0) + (x[1] - y0)*(x[1] - y0) ) )', 
                           a = - np.log(radiusThreshold)/ri**2, x0 = xi, y0 = yi, degree = 2)


E = 10.0
nu = 0.3

mu = elut.eng2mu(nu,E)
lamb = elut.eng2lambPlane(nu,E)

class OnBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

eps_ = np.array([[1.0,0.1],[0.1,.8]])

eps = as_matrix(eps_)
epsT = as_matrix(B@eps_@B.T)

onBoundary = OnBoundary()

def sigma(u):
    return lamb*nabla_div(u)*Identity(2) + 2*mu*fela.epsilon(u)

u = TrialFunction(Vref)
v = TestFunction(Vref)
a = inner(fac*sigma(u),fela.epsilon(v))*dx
aT = inner(facT*sigma(u),fela.epsilon(v))*dx

f = -inner(fac*sigma(eps*x) , fela.epsilon(v))*dx
fT = -inner(facT*sigma(epsT*x), fela.epsilon(v))*dx

bcs = DirichletBC(Vref, Constant((0.,0.)), onBoundary)


A, b = assemble_system(a, f, bcs)    
sol = Function(Vref)
solve(A, sol.vector(), b)
    
AT, bT = assemble_system(aT, fT, bcs)    
solT = Function(Vref)
solve(AT, solT.vector(), bT)


Bmultiplication = feut.affineTransformationExpession(np.zeros(2), B, Mref)
solT_comp = interpolate( feut.myfog_expression(Bmultiplication, feut.myfog(sol,Finv)), Vref) # 
ee = solT - solT_comp
error = assemble(inner(ee,ee)*dx)

print(error)

plt.figure(1)
plot(solT_comp[0]) 

plt.figure(2)
plot(solT_comp[1]) 

plt.figure(3)
plot(solT[0]) 

plt.figure(4)
plot(solT[1])

Vref0 = FunctionSpace(Mref,"CG", 1) 
facProj = project(fac, Vref0)
facTProj = project(facT, Vref0)

plt.figure(5)
plot(facProj) 

plt.figure(6)
plot(facTProj)
