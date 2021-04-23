import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0,'../utils/')

# import generation_deepBoundary_lib as gdb
import matplotlib.pyplot as plt
import numpy as np

import myTensorflow as mytf
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
# from mpi4py import MPI

# comm = MPI.COMM_WORLD

# Test Loading 

folder = './comparisonSimulations/'

# Lx = 2.0
# Ly = 0.5
# Nx = 10
# Ny = 4
# ty = -0.01
# mesh = RectangleMesh(comm,Point(0.0, 0.0), Point(Lx, Ly), Nx, Ny, "right/left") # needed to recreate since reading from file was wrong

# Create mesh and define function space


tangent_labels = ['dnn_big_140', 'dnn_big_40', 'dnn_medium_40', 'per', 'MR', 'lin']
files = [ 'barMacro_full.xdmf', 'barMacro_reduced_dnn_big_140.xdmf', 'barMacro_reduced_dnn_big_40.xdmf', 'barMacro_reduced_dnn_medium_40.xdmf', 
          'barMacro_reduced_per.xdmf', 'barMacro_reduced_MR.xdmf', 'barMacro_reduced_lin.xdmf']

files = [folder + f for f in files]

meshFile = folder + 'meshBarMacro.xdmf'
mesh = Mesh()
with XDMFFile(meshFile) as infile:
        infile.read(mesh)
        
Uh = VectorFunctionSpace(mesh, "CG", 1)
meshes = []
uhs = []
for f in files:    
    uh = Function(Uh)
    with XDMFFile(f) as infile:
        infile.read_checkpoint(uh, 'u', 0)
        uhs.append(uh)
        
pA = Point(2.0,0.0)
pB = Point(2.0,0.5)

norms_label = ['L2', '||u(A)||', '||u(B)||', '|u_1(A)|', '|u_1(B)|', '|u_2(A)|', '|u_2(B)|']
norms = [lambda x: norm(x), lambda x: np.linalg.norm(x(pA)), lambda x: np.linalg.norm(x(pB)), 
         lambda x: abs(x(pA)[0]), lambda x: abs(x(pB)[0]), lambda x: abs(x(pA)[1]), lambda x: abs(x(pB)[1])]
norms_ref = [N(uhs[0]) for N in norms] 

norms = norms + [lambda x: N(x)/norms_ref[i] for i, N in enumerate(norms)]

e = Function(Uh)
n = len(files) - 1
errors = np.zeros((n,len(norms)))

for i in range(n):
    e.vector().set_local(uhs[i+1].vector().get_local()[:]-uhs[0].vector().get_local()[:])
    for j, N in enumerate(norms):
        errors[i,j] = N(e)
 
np.savetxt(folder + 'errors.txt', errors)

folderImages = './comparisonSimulations/'
suptitle = 'Error_bar_vertical_load_on_tip' 

plt.figure(1)
plt.title(suptitle)
for i in range(6):
    plt.plot(errors[i,0::2], '-o', label = tangent_labels[i])
plt.yscale('log')
plt.xticks([0,1,2,3,4,5,6],labels = norms_label)
plt.xlabel('Norms')
plt.grid()
plt.legend(loc = 'best')
plt.savefig(folderImages + suptitle + '.png')

suptitle = 'Error_rel_bar_vertical_load_on_tip' 

plt.figure(2)
plt.title(suptitle)
for i in range(6):
    plt.plot(errors[i,1::2], '-o', label = tangent_labels[i])
plt.yscale('log')
plt.xticks([0,1,2,3,4,5,6],labels = norms_label)
plt.xlabel('Relative Norms')
plt.grid()
plt.legend(loc = 'best')
plt.savefig(folderImages + suptitle + '.png')

