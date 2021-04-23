import sys, os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import dolfin as df 
import matplotlib.pyplot as plt
from ufl import nabla_div
sys.path.insert(0, '/home/felipefr/github/micmacsFenics/utils/')
sys.path.insert(0,'../../utils/')

import multiscaleModels as mscm
from fenicsUtils import symgrad, symgrad_voigt, Integral
import numpy as np

import fenicsMultiscale as fmts
import myHDF5 as myhd
import meshUtils as meut
import elasticity_utils as elut
import symmetryLib as symlpy
from timeit import default_timer as timer
import multiphenics as mp
from MicroConstitutiveModelDNN import *

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_self = MPI.COMM_SELF
rank = comm.Get_rank()
num_ranks = comm.Get_size()

volFrac = ''
folderTangent = './tangents{0}/'.format(volFrac)
folderMesh = './meshes{0}/'.format(volFrac)
model = 'dnn'
modelBnd = 'lin'
if(model == 'dnn'):
    modelDNN = '_big_140' # underscore included before
    BCname = folderTangent + 'BCsPrediction_RVEs.hd5'
else:
    modelDNN = ''
    
start = timer()
    
# loading boundary reference mesh
nameMeshRefBnd = '../boundaryMesh.xdmf'
Mref = meut.EnrichedMesh(nameMeshRefBnd,comm_self)
Vref = df.VectorFunctionSpace(Mref,"CG", 1)

dxRef = df.Measure('dx', Mref) 

# defining the micro model

ellipseDataName = 'ellipseData_RVEs{0}.hd5'.format(volFrac)
Centers = myhd.loadhd5(ellipseDataName, 'center')[rank::num_ranks,:]

ns = len(Centers) 
nCells = ns*num_ranks # let's suppose num_ranks exact divisor

tangentFile = folderTangent + 'tangent_{0}_{1}.hd5'.format(model,rank)
os.system('rm ' + tangentFile)
Iid_tangent_center, f = myhd.zeros_openFile(tangentFile, [(ns,), (ns,3,3), (ns,2)],
                                       ['id', 'tangent','center'], mode = 'w')

Iid, Itangent, Icenter = Iid_tangent_center
k = 0



if(model == 'dnn'):
    u0_p = myhd.loadhd5(BCname, 'u0')[rank::num_ranks,:]
    u1_p = myhd.loadhd5(BCname, 'u1')[rank::num_ranks,:]
    u2_p = myhd.loadhd5(BCname, 'u2')[rank::num_ranks,:]

for i in range(nCells):
    if(i%num_ranks == rank):
        contrast = 10.0
        E2 = 1.0
        nu = 0.3
        param = [nu,E2*contrast,nu,E2]
        print(i)
        meshMicroName = folderMesh + 'mesh_micro_{0}_reduced.xml'.format(i)

        microModel = MicroConstitutiveModelDNN(meshMicroName, param, modelBnd) 
        if(model == 'dnn'):
            microModel.others['uD'] = df.Function(Vref) 
            microModel.others['uD0_'] = u0_p[k]
            microModel.others['uD1_'] = u1_p[k]
            microModel.others['uD2_'] = u2_p[k]
        elif(model == 'lin'):
            microModel.others['uD'] = df.Function(Vref) 
            microModel.others['uD0_'] = np.zeros(Vref.dim())
            microModel.others['uD1_'] = np.zeros(Vref.dim())
            microModel.others['uD2_'] = np.zeros(Vref.dim())
            
        
        Iid[k] = i
        Itangent[k,:,:] = microModel.getTangent()
        k = k + 1

f.close()



comm.Barrier()

if(rank == 0):
    tangentFile = folderTangent + 'tangent_dnn_{0}_{1}.hd5'
    tangentFileMerged = folderTangent + 'tangent_dnn_{0}.hd5'.format(model)
    os.system('rm ' + tangentFileMerged)
    myhd.merge([tangentFile.format(model,i) for i in range(num_ranks)], tangentFileMerged, 
                InputLabels = ['id','tangent', 'center'], OutputLabels = ['id','tangent', 'center'], axis = 0, mode = 'w-')
    
    [os.system('rm ' + tangentFile.format(model,i)) for i in range(num_ranks)]
    
    end = timer()
    np.savetxt("time_elapsed_reduced_{0}.txt".format(model), np.array([end-start]))
