from __future__ import print_function
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '../utils/')
from timeit import default_timer as timer
import elasticity_utils as elut
import ioFenicsWrappers as iofe
import Snapshots as snap
import Generator as gene
import genericParam as gpar
import fenicsWrapperElasticity as fela
import multiscale as mts
import myCoeffClass as coef

# get_ipython().magic(u'matplotlib notebook')

a = 1.         # unit cell width
b = 1.0 # unit cell height
c = 0.0        # horizontal offset of top boundary
R = 0.2        # inclusion radius
vol = a*b      # unit cell volume
# we define the unit cell vertices coordinates for later use
vertices = np.array([[0, 0.],
                     [a, 0.],
                     [a+c, b],
                     [c, b]])
fname = "./periodic_homog_elas/hexag_incl"

# if I have .geo

d = {'r' : 0.2, 'x0' : 0.5, 'y0' : 0.5, 'lc' : 0.05, 'N' : 20}
# d = iofe.getDefaultParameters()

iofe.exportMeshXML(d, referenceGeo = 'BasicGeo.geo', meshFile = 'mesh.xml')

femData = { 
            # 'defaultMeshParam': defaultMeshParam , 
            # 'problem' : lambda x,y: fela.solveElasticityBimaterial_twoBCs_2(x, y, traction).vector().get_local(),
           'fespace' : {'spaceType' : 'V', 'name' : 'u', 'spaceFamily' : 'CG', 'degree' : 2} }

mesh = fela.EnrichedMesh('mesh.xml')
mesh.createFiniteSpace(**femData['fespace'])


maxBoundaries = np.max(mesh.boundaries.array().astype('int32')) + 1

subd = mesh.subdomains.array().astype('int32') - 5 # to give the right range
mesh.dx = Measure('dx', domain=mesh, subdomain_data=subd)

Em = 50e3
num = 0.2
Er = 210e3
nur = 0.3
material_parameters = np.array([ [elut.eng2lambPlane(num,Em), elut.eng2mu(num,Em)], [elut.eng2lambPlane(nur,Er) , elut.eng2mu(nur,Er)] ])

# lame = coef.getMyCoeff(materials, param, op = 'cpp')

def eps(v):
    return sym(grad(v))

# def sigma(u, Eps):
#     return lame[0]*tr(Eps + eps(u))*Identity(2) + 2*lame[1]*(eps(u) + Eps)

nphases = len(material_parameters)


def sigma(v, i, Eps):
    E, nu = material_parameters[i]
    lmbda = E*nu/(1+nu)/(1-2*nu)
    mu = E/2/(1+nu)
    return lmbda*tr(eps(v) + Eps)*Identity(2) + 2*mu*(eps(v)+Eps)

Ve = VectorElement("CG", mesh.ufl_cell(), 2)
Re = VectorElement("R", mesh.ufl_cell(), 0)
W = FunctionSpace(mesh, MixedElement([Ve, Re]), constrained_domain=mts.PeriodicBoundary(vertices, tolerance=1e-10))
V = FunctionSpace(mesh, Ve)


v_,lamb_ = TestFunctions(W)
dv, dlamb = TrialFunctions(W)
w = Function(W)

Eps = Constant(((0, 0), (0, 0)))
# F = inner(sigma(dv, Eps), eps(v_))*mesh.dx
F = sum([inner(sigma(dv, i, Eps), eps(v_))*mesh.dx(i) for i in range(nphases)])
a, L = lhs(F), rhs(F)
a += dot(lamb_,dv)*mesh.dx + dot(dlamb,v_)*mesh.dx


def macro_strain(k):
    Eps = np.zeros((2,2))
    i , j = int(k/2) , k%2
    Eps[i,j] = 1.0
    return 0.5*(Eps + Eps.T) 

def stress2Voigt(s):
    return as_vector([s[0,0], s[1,1], s[0,1]])


# Homogenisation of Tangent
# Chom = np.zeros((2, 2, 2, 2))
# for (kk, case) in enumerate(["Exx", "Exy", "Eyx", "Eyy"]):
#     print("Solving {} case...".format(case))
#     Eps.assign(Constant(macro_strain(kk)))
#     solve(a == L, w, [], solver_parameters={"linear_solver": "cg"})
#     (v, lamb) = split(w)
#     Sigma = np.zeros((2,2))
    
#     for i in range(2):
#         for j in range(2):
#             Sigma[i,j] = assemble(sum([sigma(v, ii, Eps)[i,j]*mesh.dx(ii) for ii in range(nphases)]))/vol
    
    
#     k , l = int(kk/2) , kk%2
    
#     Chom[:,:,k,l] = Sigma

#     outputFile = "sol_" + case + ".xdmf"
#     utot = project( dot(Eps, Expression(("x[0]","x[1]"), degree=1)) + v , V)
#     iofe.postProcessing_simple(utot, outputFile)

# print(np.array_str(Chom, precision=2))

# Homogenisation of stress
nt = 10
sigHom = np.zeros((nt,2,2))
imposedEps = lambda t : np.array([[t,0],[0,0]])
 
for (kk,t) in enumerate(np.linspace(0.0,1.0,nt)):
    Eps.assign(Constant(imposedEps(t)))
    solve(a == L, w, [], solver_parameters={"linear_solver": "mumps"})
    (v, lamb) = split(w)
    Sigma = np.zeros((2,2))
    
    # for i in range(2):
    #     for j in range(2):
    #         sigHom[kk,i,j] = assemble(sum([sigma(v, ii, Eps)[i,j]*mesh.dx(ii) for ii in range(nphases)]))/vol
    
    
    outputFile = "solStrecht" + str(kk) + ".xdmf"
    utot = project( dot(Eps, Expression(("x[0]","x[1]"), degree=1)) + v , V)
    iofe.postProcessing_simple(utot, outputFile)



# lmbda_hom = Chom[1, 1,0,0]
# mu_hom = Chom[1,0,1,0]
# print(Chom[0, 0,0,0], lmbda_hom + 2*mu_hom)


# E_hom = mu_hom*(3*lmbda_hom + 2*mu_hom)/(lmbda_hom + mu_hom)
# nu_hom = lmbda_hom/(lmbda_hom + mu_hom)/2
# print("Apparent Young modulus:", E_hom)
# print("Apparent Poisson ratio:", nu_hom)



