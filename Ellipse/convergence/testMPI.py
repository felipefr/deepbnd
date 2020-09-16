from __future__ import print_function
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
from timeit import default_timer as timer

PETScOptions.set("pc_hypre_boomeramg_print_statistics", 1)
set_log_level(LogLevel.DEBUG)

parameters["mesh_partitioner"] = "ParMETIS"
parameters["ghost_mode"] = "shared_facet"

nu0 = 0.3
E0 = 1.0

lamb0, mu0 = elut.youngPoisson2lame_planeStress(nu0, E0)

# Simulation
start = timer()

meshFile = "convergence/mesh_5.xdmf"
u = fela.solveElasticitySimple([lamb0, mu0], meshFile)
# hdf5file = HDF5File(MPI.comm_world, "solutionTestMPI_mpi.hdf5", 'w')
# hdf5file.write(u, "u")
# fela.postProcessing_simple(u, "outputTestMPI_mpi.xdmf")
    
end = timer()
print(end - start) # Time in seconds, e.g. 5.38091952400282

