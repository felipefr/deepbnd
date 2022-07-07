#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 14:21:35 2022

@author: felipe
"""

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
import deepBND.core.multiscale.misc as mtsm
from deepBND.core.mesh.degenerated_rectangle_mesh import degeneratedBoundaryRectangleMesh
from deepBND.core.multiscale.mesh_RVE import paramRVE_default

import fetricks.fenics.postprocessing.wrapper_io as iofe
import fetricks.data_manipulation.wrapper_h5py as myhd
from fetricks.fenics.mesh.mesh import Mesh

# from deepBND.creation_model.dataset.simulation_snapshots import *

def solve_snapshot(i, meshname, paramMaterial):
    
    opModel = 'per'
    microModel = mscm.MicroModel(meshname, paramMaterial, opModel)
    
    meshname_postproc = meshname[:-5] + '_to_the_paper_{0}.xdmf'.format(i)
    microModel.visualiseMicrostructure(meshname_postproc)
    

def buildSnapshots(paramMaterial, filesnames, createMesh):
    paramRVEname, meshname = filesnames
        
    ns = len(myhd.loadhd5(paramRVEname, 'param')[2135:2139,0])
         
    paramRVEdata = myhd.loadhd5(paramRVEname, 'param') 
    
    
    for i in range(ns):

        if(createMesh):
            buildRVEmesh(paramRVEdata[i,:,:], meshname, 
                         isOrdered = False, size = 'full', NxL = 2, NyL = 2, maxOffset = 2, lcar = 2/30 )
        
        solve_snapshot(i, meshname, paramMaterial)
    


if __name__ == '__main__':
    
    folder = rootDataPath + "/review2_smaller/dataset/"
    folder_mesh = folder + "/meshes_visualisation/"
    suffix = ""
    opModel = 'per'
    createMesh = True
    
    contrast = 10.0
    E2 = 1.0
    nu = 0.3
    paramMaterial = [nu,E2*contrast,nu,E2]
    
    paramRVEname = folder +  'paramRVEdataset.hd5'
    meshname = folder_mesh + "mesh_visualisation.xdmf"
    
    os.system("mkdir " + folder_mesh)
    
    filesnames = [paramRVEname, meshname]
        
    buildSnapshots(paramMaterial, filesnames, createMesh)
