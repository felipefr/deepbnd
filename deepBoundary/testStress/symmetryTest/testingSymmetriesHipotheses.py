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

# eps_3 = shear, eps_1 = axial in x, eps_2 = axial in y

# H1(alternative) : Test u(mu, eps_1) = u(mu, eps_2)  
# H2: Test u(mu, eps_3) = -u(mu, -eps_3) ==> checked

np.random.seed(3)
r0 = 0.1
r1 = 0.4
NxL = NyL = 2
x0 = y0 = -1.0
Lyt = Lxt = 2.0
maxOffset = 2

nxy = 20
Mref = RectangleMesh(Point(x0, y0), Point(x0+Lxt, y0+Lyt), nxy, nxy, "right/left")
Vref = VectorFunctionSpace(Mref,"CG", 2)

x = SpatialCoordinate(Mref)
    
N = (2*maxOffset + NxL)*(2*maxOffset + NyL)

ellipseData, PermTotal, PermBox = geni.circularRegular2Regions(r0, r1, NxL, NyL, Lxt, Lyt, maxOffset, ordered = False, x0 = x0, y0 = y0)
ellipseData = ellipseData[PermTotal]

fac = Expression('1.0', degree = 2) # ground substance
radiusThreshold = 0.01

for xi, yi, ri in ellipseData[:,0:3]:
    fac = fac + Expression('exp(-a*( (x[0] - x0)*(x[0] - x0) + (x[1] - y0)*(x[1] - y0) ) )', 
                           a = - np.log(radiusThreshold)/ri**2, x0 = xi, y0 = yi, degree = 2)

E = 10.0
nu = 0.3

mu = elut.eng2mu(nu,E)
lamb = elut.eng2lambPlane(nu,E)


class OnBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

eps1_ = np.array([[1.0,0.0],[0.0,0.0]])
eps2_ = np.array([[0.0,0.0],[0.0,1.0]])
eps3_ = 0.5*np.array([[0.0,1.0],[1.0,0.0]])

# H1 testing
epsL = as_matrix(eps1_)
epsR = as_matrix(eps2_)

# H2 testing
# epsL = as_matrix(eps3_)
# epsR = as_matrix(-eps3_)

onBoundary = OnBoundary()

def sigma(u):
    return lamb*nabla_div(u)*Identity(2) + 2*mu*fela.epsilon(u)

u = TrialFunction(Vref)
v = TestFunction(Vref)
a = inner(fac*sigma(u),fela.epsilon(v))*dx

fL = -inner(fac*sigma(epsL*x) , fela.epsilon(v))*dx
fR = -inner(fac*sigma(epsR*x), fela.epsilon(v))*dx

bcs = DirichletBC(Vref, Constant((0.,0.)), onBoundary)

AL, bL = assemble_system(a, fL, bcs)    

AR, bR = assemble_system(a, fR, bcs)    


solL = Function(Vref)
solve(AL, solL.vector(), bL)

solR = Function(Vref)
solve(AR, solR.vector(), bR)

# solR.vector().set_local(-solR.vector().get_local()[:]) # to test H2

theta = np.pi/2.0
s = np.sin(theta)
c = np.cos(theta)
B = np.array([[c,-s],[s,c]]).T

# Finv = feut.affineTransformationExpession(np.zeros(2), B.T, Mref)
# Bmultiplication = feut.affineTransformationExpession(np.zeros(2), B, Mref)
# solR_comp = interpolate( feut.myfog_expression(Bmultiplication, solR), Vref ) # 

ee = solL - solR
error = assemble(inner(ee,ee)*dx)

print(error)

plt.figure(1)
plot(solL[0]) 

plt.figure(2)
plot(solL[1]) 

plt.figure(3)
plot(solR[0]) 

plt.figure(4)
plot(solR[1])

Vref0 = FunctionSpace(Mref,"CG", 1) 
facProj = project(fac, Vref0)

plt.figure(5)
plot(facProj) 

