import os, sys
import matplotlib.pyplot as plt
import numpy as np
import dolfin as df

from deepBND.__init__ import *
import deepBND.core.data_manipulation.wrapper_h5py as myhd

# Test Loading 
problemType = ''

folder = rootDataPath + '/ellipses/cook_fresh_data_augmentation/meshes_seed%d/'
folderTangent = rootDataPath + '/ellipses/prediction_fresh_data_augmentation/'

cases = ['dnn_200', 'per_200', 'lin_200', 'full_200']
Ny_splits = [5,10,20,40]

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
listSeeds = [0,1,2,3,4,5,6,7,8,9]
# listSeeds = [0,1,2]


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

for c in cases: 
    plt.plot(Ny_splits, np.mean(D[c][:,0,:], axis = 0), '--', label = c)

plt.legend()
plt.grid()
plt.savefig('dispA.eps')
plt.savefig('dispA.pdf')


E = {}


E['dnn_200'] = np.abs(D['dnn_200'] - D['full_200'])/D['full_200']
E['per_200'] = np.abs(D['per_200'] - D['full_200'])/D['full_200']

for i in range(4):
    print("%e \pm %e & %e \pm %e "%(np.mean(E['dnn_200'][:,i,-2]), np.std(E['dnn_200'][:,i,-2]),
                                    np.mean(E['per_200'][:,i,-2]), np.std(E['per_200'][:,i,-2]) ) )

# plt.figure(2)
# plt.title('Error Sigma Von Mises D')
# plt.plot(Ny_splits, np.mean(E['dnn_200'][:,1,:], axis = 0), '-', label='dnn_200')
# plt.plot(Ny_splits, np.mean(E['per_200'][:,1,:], axis = 0), '-', label='reducedper')
# plt.yscale('log')
# plt.legend()
# # plt.savefig('sigD_1_error.png')
# plt.grid()




# ========= Error in the tangent ======================================

tangentName = folderTangent + 'tangents_{0}.hd5'
tangents = {}
for case in cases:
    tangents[case] = myhd.loadhd5(tangentName.format(case), 'tangent')


refCase = 'full_200'
errors = {}
ns = 200
for case in cases[:-1]:
    errors[case] = np.zeros(ns)
    for i in range(ns):
        errors[case][i] = np.linalg.norm(tangents[case][i] - tangents[refCase][i])
        
for case in cases[:-1]:
    print(case, np.mean(errors[case]), np.std(errors[case]))
