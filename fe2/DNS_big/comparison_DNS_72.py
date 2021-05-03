import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0,'../../utils/')

# import generation_deepBoundary_lib as gdb
import matplotlib.pyplot as plt
import numpy as np

# import myTensorflow as mytf
from timeit import default_timer as timer

import h5py
import pickle
import myHDF5 
import dataManipMisc as dman 
from sklearn.preprocessing import MinMaxScaler
import miscPosprocessingTools as mpt
import myHDF5 as myhd

import matplotlib.pyplot as plt
import symmetryLib as syml
import tensorflow as tf
import meshUtils as meut
from dolfin import *
from mpi4py import MPI

# comm = MPI.COMM_WORLD

# Test Loading 

folder = './DNS_72/'
jump = '_jump'
Ny_mesh = '48'

# Lx = 2.0
# Ly = 0.5
# Nx = 10
# Ny = 4
# ty = -0.01
# mesh = RectangleMesh(comm,Point(0.0, 0.0), Point(Lx, Ly), Nx, Ny, "right/left") # needed to recreate since reading from file was wrong

# Create mesh and define function space


tangent_labels = ['DNS', 'dnn', 'full_per']
solutions = [ 'barMacro_DNS_P2_interp_ny{0}.xdmf'.format(Ny_mesh), 'multiscale{0}/barMacro_Multiscale_dnn_ny{1}.xdmf'.format(jump,Ny_mesh), 
          'multiscale{0}/barMacro_Multiscale_full_per_ny{1}.xdmf'.format(jump, Ny_mesh)] # temporary
         
solutions = [folder + f for f in solutions]

meshDNSfile =  folder + 'mesh.xdmf'
meshMultiscaleFile = folder + 'multiscale{0}/meshBarMacro_Multiscale_ny{1}.xdmf'.format(jump,Ny_mesh)

meshMultiscale = Mesh()
with XDMFFile(meshMultiscaleFile) as infile:
    infile.read(meshMultiscale)
        

Uh_mult = VectorFunctionSpace(meshMultiscale, "CG", 2)

uhs = []
for f in solutions:    
    uh = Function(Uh_mult)
    with XDMFFile(f) as infile:
        infile.read_checkpoint(uh, 'u', 0)
        uhs.append(uh)

# uhs[0] = interpolate(uhs[0],Uh_mult)

pA = Point(2.0,0.0)
pB = Point(2.0,0.5)

norms_label = ['L2', '||u(A)||', '||u(B)||', '|u_1(A)|', '|u_1(B)|', '|u_2(A)|', '|u_2(B)|']
norms = [lambda x: norm(x), lambda x: np.linalg.norm(x(pA)), lambda x: np.linalg.norm(x(pB)), 
          lambda x: abs(x(pA)[0]), lambda x: abs(x(pB)[0]), lambda x: abs(x(pA)[1]), lambda x: abs(x(pB)[1])]
norms_ref = [N(uhs[0]) for N in norms] 

norms = norms + [lambda x: N(x)/norms_ref[i] for i, N in enumerate(norms)]

e = Function(Uh_mult)
n = len(solutions) - 1
errors = np.zeros((n,len(norms)))

for i in range(n):
    e.vector().set_local(uhs[i+1].vector().get_local()[:]-uhs[0].vector().get_local()[:])
    for j, N in enumerate(norms):
        errors[i,j] = N(e)
 
# np.savetxt(folder + 'errors.txt', errors)

suptitle = 'Error_DNS_72_bar_nyMesh{1}_vs_DNS{0}'.format(jump,Ny_mesh) 

np.savetxt(folder + 'errors_{0}.txt'.format(suptitle), errors)

plt.figure(1)
plt.title(suptitle)
for i in range(n):
    plt.plot(errors[i,0::2], '-o', label = tangent_labels[i+1]) # jump tangent_label 0 (reference)
plt.yscale('log')
plt.xticks([0,1,2,3,4,5,6],labels = norms_label)
plt.xlabel('Norms')
plt.grid()
plt.legend(loc = 'best')
plt.savefig(folder + suptitle + '.png')

suptitle = 'Error_DNS_72_rel_nyMesh{1}_vs_DNS{0}'.format(jump, Ny_mesh) 

plt.figure(2)
plt.title(suptitle)
for i in range(n):
    plt.plot(errors[i,1::2], '-o', label = tangent_labels[i+1]) # jump tangent_label 0 (reference)
plt.yscale('log')
plt.xticks([0,1,2,3,4,5,6],labels = norms_label)
plt.xlabel('Relative Norms')
plt.grid()
plt.legend(loc = 'best')
plt.savefig(folder + suptitle + '.png')

