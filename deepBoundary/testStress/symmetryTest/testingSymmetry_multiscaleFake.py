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

E = 10.0
nu = 0.3

mu = elut.eng2mu(nu,E)
lamb = elut.eng2lambPlane(nu,E)
fac = 0.1


class OnBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

s0 = -1.0
s1 = -1.0
J = as_matrix(((s0,0.0),(0.0,s1)))
T = Expression(('s0*x[0]','s1*x[1]'), s0=s0, s1=s1, degree = 2)

eps = as_matrix(((0.0,0.5),(0.5,0.0)))
epsT = J*eps*J

onBoundary = OnBoundary()

x0 = y0 = -1.0
Lyt = Lxt = 2.0
nxy = 100
Mref = RectangleMesh(Point(x0, y0), Point(x0+Lxt, y0+Lyt), nxy, nxy, "right/left")
Vref = VectorFunctionSpace(Mref,"CG", 2)

x = SpatialCoordinate(Mref)

def sigma(u):
    return (lamb + fac*x[0])*nabla_div(u)*Identity(2) + 2*(mu - fac*x[1])*fela.epsilon(u)

def sigmaT(u):
    return (lamb + s0*fac*x[0])*nabla_div(u)*Identity(2) + 2*(mu - s1*fac*x[1])*fela.epsilon(u)

u = TrialFunction(Vref)
v = TestFunction(Vref)
a = inner(sigma(u),fela.epsilon(v))*dx
aT = inner(sigmaT(u),fela.epsilon(v))*dx

f = -inner(sigma(eps*x) , fela.epsilon(v))*dx
fT = -inner(sigmaT(epsT*x), fela.epsilon(v))*dx

bcs = DirichletBC(Vref, Constant((0.,0.)), onBoundary)


A, b = assemble_system(a, f, bcs)    
sol = Function(Vref)
solve(A, sol.vector(), b)
    
A, b = assemble_system(aT, fT, bcs)    
solT = Function(Vref)
solve(A, solT.vector(), b)

solT_comp = J*feut.myfog(sol,T)
ee = solT - solT_comp
error = assemble(inner(ee,ee)*dx)

plt.figure(1)
plot((eps*x + sol)[0]) 

plt.figure(2)
plot((eps*x + sol)[1]) 

plt.figure(3)
plot((epsT*x + solT)[0]) 

plt.figure(4)
plot((epsT*x + solT)[1])
