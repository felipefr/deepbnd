import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0,'../../utils/')

# import generation_deepBoundary_lib as gdb
import matplotlib.pyplot as plt
import numpy as np

# import myTensorflow as mytf
from timeit import default_timer as timer

import h5py
import myHDF5 as myhd
import matplotlib.pyplot as plt
import meshUtils as meut
from dolfin import *
import plotUtils

# Test Loading 

rootDataPath = open('../../../rootDataPath.txt','r').readline()[:-1]
print(rootDataPath)

Ny_DNS = 72
problemType = ''

folder = rootDataPath + '/new_fe2/cook/meshes_%d/'

cases = ['full', 'dnn', 'reduced_per']
Ny_splits = [5,10,20,40,80]

solution = folder + 'cook_%s_%d.xdmf'
meshfile =  folder + 'mesh_%d.xdmf'

pA = Point(48.0,60.0)
pB = Point(35.2,44.0)
pC = Point(24.0,22.1)
pD = Point(16.0,49.0)

norms_label = ['uyA', 'sigB', 'sigC', 'sigD']
norms = [lambda x,y: x(pA)[1], lambda x,y: y(pB)[0], lambda x,y: y(pC)[0], lambda x,y: y(pD)[0]]

uhs = []
vonMises_ = []
# listSeeds = [0,1,2,3,6,7,8,9,10]
listSeeds = [0,1,2,3,4,5,6,7,8,9,10]


D = {}
for case in cases:
    D[case] = np.zeros((len(listSeeds),len(norms),len(Ny_splits)))


for kk, k in enumerate(listSeeds):
    for i, ny in enumerate(Ny_splits):    
    
        mesh = Mesh()
        with XDMFFile(meshfile%(k,ny)) as infile:
            infile.read(mesh)
    
        Uh = VectorFunctionSpace(mesh, "CG", 2)
        DG0 = VectorFunctionSpace(mesh, "DG", 0)
        
        uh = Function(Uh)
        vonMises = Function(DG0)
    
        for case in cases:
            
            with XDMFFile(solution%(k,case,ny)) as infile:
                infile.read_checkpoint(uh, 'u', 0)
                infile.read_checkpoint(vonMises, 'vonMises', 0)
            
            for j, norm in enumerate(norms):
                D[case][kk,j,i] = norm(uh,vonMises)     
                




plt.figure(1)
plt.title('Vertical Tip Displacement (A)')
plt.ylabel("\Large $\mathbf{u}_{2}$")
plt.xlabel("Number of vertical elements")
plt.plot(Ny_splits, np.mean(D['full'][:,0,:], axis = 0), '--', label='High-Fidelity')
plt.plot(Ny_splits, np.mean(D['dnn'][:,0,:], axis = 0), 'o', label='DeepBND')
plt.plot(Ny_splits, np.mean(D['reduced_per'][:,0,:], axis = 0), 'x', label="Periodic")
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
