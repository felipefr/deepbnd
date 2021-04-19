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
from mpi4py import MPI

comm = MPI.COMM_WORLD

# Test Loading 

folder = './comparisonSimulations/'

Lx = 2.0
Ly = 0.5
Nx = 10
Ny = 4
ty = -0.01
mesh = RectangleMesh(comm,Point(0.0, 0.0), Point(Lx, Ly), Nx, Ny, "right/left") # needed to recreate since reading from file was wrong

# Create mesh and define function space

files = [ 'barMultiscale_per_full.xdmf', 'barMultiscale_lin.xdmf', 'barMultiscale_per.xdmf', 
          'barMultiscale_MR.xdmf', 'barMultiscale_dnn.xdmf']

files = [folder + f for f in files]

meshFile = folder + 'meshBarMacro.xdmf'

meshes = []
uhs = []
for f in files:    
    Uh = VectorFunctionSpace(mesh, "Lagrange", 1)
    uh = Function(Uh)
    with XDMFFile(comm,f) as infile:
        infile.read_checkpoint(uh, 'u', 0)
        uhs.append(uh)
        
        
n = len(files) - 1
errors = np.zeros((n,4))

uref_L2 = norm(uhs[0])
uref_tipNorm =np.linalg.norm(uhs[0](Point(2.0,0.0)))
for i in range(n):
    errors[i,0] = errornorm(uhs[i+1],uhs[0])
    errors[i,1] = np.linalg.norm(uhs[i+1](Point(2.0,0.0)) - uhs[0](Point(2.0,0.0)) )
    errors[i,2] = errors[i,0]/uref_L2
    errors[i,3] = errors[i,1]/uref_tipNorm        

np.savetxt(folder + 'errors.txt', errors)

# def load_basicStructureU(self):
     # u = Function(self.Mesh.V['u'])
     # with HDF5File(MPI.comm_world, self.solutionFile, 'r') as f:
         # f.read(u, 'basic')
        # 
     # return u

# folderTrain = './models/dataset_shear1/'
# folderBasis = './models/dataset_shear1/'

# NlistModel = [5,40,140]
# predict_file = 'sigma_prediction_ny{0}.hd5'
# predict_file_shear = 'sigma_prediction_ny{0}_shear.hd5'

# ns = 100
# sigma = []
# sigma_ref = []
# error = np.zeros((len(NlistModel), ns))
# error_shear = np.zeros((len(NlistModel), ns))

# for j, jj in enumerate(NlistModel): 
#     error[j,:] =  myhd.loadhd5(predict_file.format(jj), 'error').flatten()**2
#     error_shear[j,:] =  myhd.loadhd5(predict_file_shear.format(jj), 'error').flatten()**2



# # # ================ Alternative Prediction ========================================== 
# folderImages = './images/'
# suptitle = 'Error_Stress_Pure_Axial_Shear_Square.{0}' 

# plt.figure(1)
# plt.title(suptitle.format('', ''))
# plt.plot(NlistModel, np.mean(error, axis = 1), '-o', label = 'mean (axial)')
# plt.plot(NlistModel, np.mean(error, axis = 1) + np.std(error, axis = 1) , '--', label = 'mean + std (axial)')
# plt.plot(NlistModel, np.mean(error_shear, axis = 1), '-o', label = 'mean (shear)')
# plt.plot(NlistModel, np.mean(error_shear, axis = 1) + np.std(error, axis = 1) , '--', label = 'mean + std (shear)')
# # plt.ylim(1.0e-8,1.0e-5)
# plt.yscale('log')
# plt.xlabel('N')
# plt.ylabel('Squared Error homogenised stress Frobenius')
# plt.grid()
# plt.legend(loc = 'best')
# # plt.savefig(suptitle.format('png'))
