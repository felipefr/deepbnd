"""
This file is part of deepBND, a data-driven enhanced boundary condition implementaion for 
computational homogenization problems, using RB-ROM and Neural Networks.
Copyright (c) 2020-2023, Felipe Rocha.
See file LICENSE.txt for license information. 
Please cite this work according to README.md.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or <felipe.f.rocha@gmail.com>
"""

import sys, os
import numpy as np
import dolfin as df

from deepBND.__init__ import *

from fetricks.fenics.mesh.mesh import Mesh 
import fetricks.data_manipulation.wrapper_h5py as myhd


folder = rootDataPath + "/review2_smaller/dataset_coarsed/"
    

snapshots_fine = folder + 'snapshots.hd5'
snapshots_coarse = folder + 'snapshots_coarse.hd5' 

mesh_fine_name =  folder + 'mesh_fine/boundaryMesh.xdmf'
mesh_coarse_name =  folder + 'boundaryMesh.xdmf'


meshFine = Mesh(mesh_fine_name)
meshCoarse = Mesh(mesh_coarse_name)

Uh_fine = df.VectorFunctionSpace(meshFine, "CG", 2)
Uh_coarse = df.VectorFunctionSpace(meshCoarse, "CG", 1)

labels = ['id', 'solutions_S','sigma_S','a_S','B_S', 'sigmaTotal_S',
         'solutions_A','sigma_A','a_A','B_A', 'sigmaTotal_A']
data, fdata = myhd.loadhd5_openFile(snapshots_fine, labels)

sol = [ data[1] , data[6] ]

ns = data[0].shape[0]
Nh_coarse = Uh_coarse.dim()
sol_new = [ np.zeros((ns,Nh_coarse)), np.zeros((ns,Nh_coarse))]

uh_fine = df.Function(Uh_fine)
uh_coarse = df.Function(Uh_coarse)

for k in range(len(sol)):
    for i in range(ns):
        print("interpolating sol i", i)
        uh_fine.vector().set_local( sol[k][i,:] )
        uh_coarse.interpolate(uh_fine)
        sol_new[k][i,:] = uh_coarse.vector().get_local()[:]
        
data[1] = sol_new[0]
data[6] = sol_new[1]
 
snapshots = myhd.savehd5( snapshots_coarse, data,
                            label = labels, mode = 'w')
    
fdata.close()

# meshDNS = meut.EnrichedMesh(meshDNSfile)
# with XDMFFile(meshDNSfile) as infile:
#     infile.read(meshDNS)

# print("mesh DNS loaded")

# meshMultiscale = Mesh()
# with XDMFFile(meshMultiscaleFile) as infile:
#     infile.read(meshMultiscale)
        

# Uh_DNS = VectorFunctionSpace(meshDNS, "CG", 2)
# print(Uh_DNS.dim())
# print("space DNS created")
# Uh_mult = VectorFunctionSpace(meshMultiscale, "CG", 2)
# print(Uh_mult.dim())

# uh_DNS = Function(Uh_DNS)
# with XDMFFile(solution_DNS) as infile:
#     infile.read_checkpoint(uh_DNS, 'u', 0)

# uh_DNS_interp = interpolate(uh_DNS,Uh_mult)

# with XDMFFile("barMacro_DNS.xdmf") as file:
#     file.write_checkpoint(uh_DNS_interp,'u',0)
    