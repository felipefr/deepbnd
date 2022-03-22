import os, sys
import matplotlib.pyplot as plt
import numpy as np
import dolfin as df

from deepBND.__init__ import *
import deepBND.core.data_manipulation.wrapper_h5py as myhd

# Test Loading 
problemType = ''

folder = rootDataPath + '/circles/cook/meshes_seed%d/'

cases = ['full', 'dnn', 'reduced_per']
Ny_splits = [5,10,20,40,80]

solution = folder + 'cook_%s_%d.xdmf'
meshfile =  folder + 'mesh_%d.xdmf'

pA = df.Point(48.0,60.0)
pB = df.Point(35.2,44.0)
pC = df.Point(24.0,22.1)
pD = df.Point(16.0,49.0)

norms_label = ['uyA', 'sigB', 'sigC', 'sigD']
norms = [lambda x,y: x(pA)[1], lambda x,y: y(pB)[0], lambda x,y: y(pC)[0], lambda x,y: y(pD)[0]]

uhs = []
vonMises_ = []
# listSeeds = [0,1,2,3,6,7,8,9,10]
listSeeds = [0,1,2]


D = {}
for case in cases:
    D[case] = np.zeros((len(listSeeds),len(norms),len(Ny_splits)))


for kk, k in enumerate(listSeeds):
    for i, ny in enumerate(Ny_splits):    
    
        mesh = df.Mesh()
        with df.XDMFFile(meshfile%(k,ny)) as infile:
            infile.read(mesh)
    
        Uh = df.VectorFunctionSpace(mesh, "CG", 2)
        DG0 = df.VectorFunctionSpace(mesh, "DG", 0)
        
        uh = df.Function(Uh)
        vonMises = df.Function(DG0)
    
        for case in cases:
            
            with df.XDMFFile(solution%(k,case,ny)) as infile:
                infile.read_checkpoint(uh, 'u', 0)
                infile.read_checkpoint(vonMises, 'vonMises', 0)
            
            for j, norm in enumerate(norms):
                D[case][kk,j,i] = norm(uh,vonMises)     
                




plt.figure(1)
plt.title('Vertical Tip Displacement (A)')
plt.ylabel("\Large $\mathbf{u}_{2}$")
plt.xlabel("Number of vertical elements")
plt.plot(Ny_splits, np.mean(D[cases[0]][:,0,:], axis = 0), '--', label='High-Fidelity')
plt.plot(Ny_splits, np.mean(D[cases[1]][:,0,:], axis = 0), 'o', label='DeepBND')
plt.plot(Ny_splits, np.mean(D[cases[2]][:,0,:], axis = 0), 'x', label="Periodic")
plt.legend()
plt.grid()
plt.savefig('dispA.eps')
plt.savefig('dispA.pdf')


E = {}

# E['dnn'] = np.abs(D['dnn'] - D['full'])/D['full']
# E['reduced_per'] = np.abs(D['reduced_per'] - D['full'])/D['full']

# for i in range(4):
#     print("%e \pm %e & %e \pm %e "%(np.mean(E['dnn'][:,i,-2]), np.std(E['dnn'][:,i,-2]),
#                                     np.mean(E['reduced_per'][:,i,-2]), np.std(E['reduced_per'][:,i,-2]) ) )

# plt.figure(2)
# plt.title('Error Sigma Von Mises D')
# plt.plot(Ny_splits, np.mean(E['dnn'][:,1,:], axis = 0), '-', label='dnn')
# plt.plot(Ny_splits, np.mean(E['reduced_per'][:,1,:], axis = 0), '-', label='reducedper')
# plt.yscale('log')
# plt.legend()
# # plt.savefig('sigD_1_error.png')
# plt.grid()
