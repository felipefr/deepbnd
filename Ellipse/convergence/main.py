from __future__ import print_function
import numpy as np
from fenics import *
from dolfin import *
from ufl import nabla_div
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, '..')
import copy

import elasticity_utils as elut

import fenicsWrapperElasticity as fela 

paramModMesh = [-0.03,0.03,0.2,0.1]


d = fela.getDefaultParameters()
# exportMeshXML(d)
# mesh = Mesh('mesh.xml')
MeshFile = 'mesh.xml'

meshReference = fela.generateParametrisedMesh(MeshFile,[0.0,0.0,0.0,0.0],d )
meshModified = fela.generateParametrisedMesh(MeshFile, paramModMesh,d )

nu0 = 0.3
nu1 = 0.3

E0 = 1.0
E1 = 100.0

lamb0, mu0 = elut.youngPoisson2lame_planeStress(nu0, E0)
lamb1, mu1 = elut.youngPoisson2lame_planeStress(nu1, E1)

u1 = fela.solveElasticityBimaterial(np.array([[lamb0, mu0], [lamb1, mu1]]), meshModified)

u2 = fela.solveElasticityBimaterial_withAffDec(np.array([[lamb0, mu0], [lamb1, mu1]]), meshReference, paramModMesh)



# V = VectorFunctionSpace(meshModified, 'CG', 2)
# u3 = Function(V)
# u3.vector().set_local(u2.vector().get_local())

# fileResults = XDMFFile("output_2.xdmf")
# fileResults.parameters["flush_output"] = True
# fileResults.parameters["functions_share_mesh"] = True

# u3.rename('u', 'displacements at nodes')    
# fileResults.write(u3,0.)

print(np.linalg.norm(u1.vector().get_local() - u2.vector().get_local()))
print(np.linalg.norm(u1.vector().get_local() - u2.vector().get_local(), np.inf))