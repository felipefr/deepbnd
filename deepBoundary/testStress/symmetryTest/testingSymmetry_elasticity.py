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

body = Expression(('sin(2*PI*x[0]) + 0.5*x[1]','cos(2*PI*x[1]) - 0.5*x[0]'), PI = np.pi , degree=2)
bodyT = Expression(('s0*sin(2*PI*s0*x[0]) + s0*s1*0.5*x[1]','s1*cos(s1*2*PI*x[1]) - s1*s0*0.5*x[0]'), PI = np.pi , s1 = s1, s0 = s0, degree=2)

f = inner(body, v)*dx
fT = inner(bodyT, v)*dx

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
plot(sol[1]) 

# T_MH = Expression(('-x[0]','x[1]'), degree = 2)
# T_MV = Expression(('x[0]','-x[1]'), degree = 2)


# Transformations = [T_MH,T_MV,T_MD]
# usol_total_mirror = [Function(Vref_total),Function(Vref_total),Function(Vref_total)]
# usol_mirror = [Function(Vref),Function(Vref),Function(Vref)]

# for i,T in enumerate(Transformations):

#     usol_mirror[i].interpolate(feut.myfog(usol[0],T)) 

# dx = Measure('dx',Mref_total)
# ds = Measure('ds', Mref)

# error_total = []
# error = []
# for i in range(3):
#     error_total.append( assemble( inner(usol_total_mirror[i] - usol_total[0], usol_total_mirror[i] - usol_total[0])*dx ) )
#     error.append( assemble( inner(usol_mirror[i] - usol[0], usol_mirror[i] - usol[0])*ds ) )
    

# plt.figure(1)
# plot(usol_total[0][1]+usol_total_mirror[2][1] , mode ='color')
# plt.colorbar()
# plt.show()

