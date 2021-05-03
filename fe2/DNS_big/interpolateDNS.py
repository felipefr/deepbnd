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

folder = './DNS_24/'

solution_DNS = folder + 'barMacro_DNS_P2.xdmf' 

meshDNSfile =  folder + 'mesh.xdmf'
meshMultiscaleFile = folder + 'multiscale/meshBarMacro_Multiscale_ny48.xdmf'

meshDNS = meut.EnrichedMesh(meshDNSfile)
print("mesh DNS loaded")

meshMultiscale = Mesh()
with XDMFFile(meshMultiscaleFile) as infile:
    infile.read(meshMultiscale)
        

Uh_DNS = VectorFunctionSpace(meshDNS, "CG", 2)
print("space DNS created")
Uh_mult = VectorFunctionSpace(meshMultiscale, "CG", 2)

uh_DNS = Function(Uh_DNS)
with XDMFFile(solution_DNS) as infile:
    infile.read_checkpoint(uh_DNS, 'u', 0)

uh_DNS_interp = interpolate(uh_DNS,Uh_mult)

with XDMFFile(folder + "barMacro_DNS_P2_interp_ny48.xdmf") as file:
    file.write_checkpoint(uh_DNS_interp,'u',0)
    