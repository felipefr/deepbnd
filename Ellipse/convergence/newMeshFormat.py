import numpy as np

import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, '..')
import copy

import os

from fenics import *
from dolfin import *
from ufl import nabla_div

import elasticity_utils as elut

import fenicsWrapperElasticity as fela 
import ioFenicsWrappers as iofe

# import ioFenicsWrappers as fela
from timeit import default_timer as timer



# Mesh construction
Folder = 'experiencesNewMeshFormat/'
MeshFileRad = 'mesh'
MeshFileExt = 'xdmf'
referenceGeo = 'reference.geo'
labelsPhysicalGroups = {'line' : 'faces', 'triangle' : 'regions'}

# d = fela.getDefaultParameters()
# hRef = 0.005
# Nref = int(d['Lx1']/hRef) + 1
# print(Nref)
# hRef = d['Lx1']/(Nref - 1)
# print(hRef)
# d['lc'] = hRef
# d['Nx1'] = Nref
# d['Nx2'] = 2*Nref - 1
# d['Nx3'] = Nref
# d['Ny1'] = Nref
# d['Ny2'] = 2*Nref - 1
# d['Ny3'] = Nref
    
# fela.exportMeshHDF5_fromGMSH(Folder + 'mesh.msh', '{0}{1}.{2}'.format(Folder, MeshFileRad, MeshFileExt), labelsPhysicalGroups)

# mesh = fela.EnrichedMesh('{0}{1}.{2}'.format(Folder, MeshFileRad, MeshFileExt))

# set_log_level(LogLevel.DEBUG)

parameters["ghost_mode"] = "shared_vertex"

nu0 = 0.3
E0 = 1.0

lamb0, mu0 = elut.youngPoisson2lame_planeStress(nu0, E0)

nu1 = 0.3
E1 = 100.0

lamb1, mu1 = elut.youngPoisson2lame_planeStress(nu1, E1)


# Simulation
start = timer()

# meshFile = '{0}{1}.{2}'.format(Folder, MeshFileRad, MeshFileExt)
mesh = fela.EnrichedMesh('{0}{1}.{2}'.format(Folder, MeshFileRad, MeshFileExt))

rank = MPI.comm_world.Get_rank()
# rank = MPI.rank
print('Outside', rank)
# u = fela.solveElasticitySimple([lamb0, mu0], meshFile)
u = fela.solveElasticityBimaterial_simpler(np.array(9*[[lamb0, mu0]] + [[lamb1,mu1]]), mesh, rank)
# hdf5file = HDF5File(MPI.comm_world, Folder + "solution.hdf5", 'w')
# hdf5file.write(u, "u")
iofe.postProcessing_simple(u, Folder + "output_bimaterial_different_mpi.xdmf", MPI.comm_world)
    
# end = timer()
# print(end - start) # Time in seconds, e.g. 5.38091952400282

