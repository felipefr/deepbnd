#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 04:46:39 2022

@author: felipe
"""

import sys, os
import numpy as np
from numpy import isclose
from dolfin import *
from multiphenics import *
import copy
from timeit import default_timer as timer
from mpi4py import MPI

from deepBND.__init__ import *
from deepBND.core.multiscale.mesh_RVE import buildRVEmesh
import deepBND.core.multiscale.micro_model as mscm
import deepBND.core.elasticity.fenics_utils as fela
import deepBND.core.fenics_tools.wrapper_io as iofe
import deepBND.core.multiscale.misc as mtsm
from deepBND.core.fenics_tools.enriched_mesh import EnrichedMesh 
import deepBND.core.data_manipulation.wrapper_h5py as myhd
from deepBND.core.mesh.degenerated_rectangle_mesh import degeneratedBoundaryRectangleMesh
from deepBND.core.multiscale.mesh_RVE import paramRVE_default

from deepBND.creation_model.dataset.simulation_snapshots import *

comm = MPI.COMM_WORLD
comm_self = MPI.COMM_SELF

if(len(sys.argv)>1):
    run = int(sys.argv[1])
    num_runs = int(sys.argv[2])
else:
    comm = MPI.COMM_WORLD
    run = comm.Get_rank()
    num_runs = comm.Get_size()
    
print('run, num_runs ', run, num_runs)

run = comm.Get_rank()
num_runs = comm.Get_size()

folder = rootDataPath + "/ellipses/dataset/"

suffix = ""
opModel = 'per'
createMesh = True

contrast = 10.0
E2 = 1.0
nu = 0.3
paramMaterial = [nu,E2*contrast,nu,E2]

bndMeshname = folder + 'boundaryMesh.xdmf'
paramRVEname = folder +  'paramRVEdataset{0}.hd5'.format(suffix)
snapshotsname = folder +  'snapshots{0}_{1}.hd5'.format(suffix,run)
meshname = folder + "meshes/mesh_temp_{0}.xdmf".format(run)

filesnames = [bndMeshname, paramRVEname, snapshotsname, meshname]

# generation of the lite mesh for the internal boundary
# p = paramRVE_default()
# meshRef = degeneratedBoundaryRectangleMesh(x0 = p.x0L, y0 = p.y0L, Lx = p.LxL , Ly = p.LyL , Nb = 21)
# meshRef.generate()
# meshRef.write(bndMeshname , 'fenics')


buildSnapshots(paramMaterial, filesnames, opModel, createMesh, run, num_runs, comm_self)