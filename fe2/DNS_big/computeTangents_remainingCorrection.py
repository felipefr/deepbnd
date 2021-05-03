import sys, os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import dolfin as df 
import matplotlib.pyplot as plt
from ufl import nabla_div
sys.path.insert(0, '/home/rocha/github/micmacsFenics/utils')
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

Ny = 72
folder = "/home/rocha/switchdrive/scratch/fe2/DNS_big/DNS_{0}/".format(Ny)
volFrac = ''
folderTangent = folder + 'tangents{0}/'.format(volFrac)
folderMesh = folder + 'meshes{0}/'.format(volFrac)
model = 'dnn'
modelBnd = 'lin'
meshSize = 'reduced'
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

ellipseDataName = folder + 'ellipseData_RVEs{0}.hd5'.format(volFrac)
size_ids = len(myhd.loadhd5(ellipseDataName, 'center')) # idMax + 1 


ids = np.loadtxt(folderTangent + "other_ids.txt").astype('int')

ids = ids[rank::num_ranks]
Centers = myhd.loadhd5(ellipseDataName, 'center')[ids,:]

ns = len(Centers) # per rank

tangentFile = folderTangent + 'tangent_{0}_{1}.hd5'.format(model,rank)
os.system('rm ' + tangentFile)
Iid_tangent_center, f = myhd.zeros_openFile(tangentFile, [(ns,), (ns,3,3), (ns,2)],
                                       ['id', 'tangent','center'], mode = 'w')

Iid, Itangent, Icenter = Iid_tangent_center

if(model == 'dnn'):
    u0_p = myhd.loadhd5(BCname, 'u0')[rank::num_ranks,:]
    u1_p = myhd.loadhd5(BCname, 'u1')[rank::num_ranks,:]
    u2_p = myhd.loadhd5(BCname, 'u2')[rank::num_ranks,:]


startTotal = timer()
for i in range(ns):
    Iid[i] = ids[i]
   
    contrast = 10.0
    E2 = 1.0
    nu = 0.3
    param = [nu,E2*contrast,nu,E2]
    
    meshMicroName = folderMesh + 'mesh_micro_{0}_{1}.xdmf'.format(int(Iid[i]), meshSize)

    microModel = MicroConstitutiveModelDNN(meshMicroName, param, modelBnd) 
    if(model == 'dnn'):
        microModel.others['uD'] = df.Function(Vref) 
        microModel.others['uD0_'] = u0_p[i] # it was already picked correctly
        microModel.others['uD1_'] = u1_p[i] 
        microModel.others['uD2_'] = u2_p[i]
    elif(model == 'lin'):
        microModel.others['uD'] = df.Function(Vref) 
        microModel.others['uD0_'] = np.zeros(Vref.dim())
        microModel.others['uD1_'] = np.zeros(Vref.dim())
        microModel.others['uD2_'] = np.zeros(Vref.dim())
        
    
    Icenter[i,:] = Centers[i,:]
    
    start = timer() 
    Itangent[i,:,:] = microModel.getTangent()
    end = timer() 
    print("time elapsed: ", end - start, i, rank )

    
    f.flush()    
    
    if(i%10 == 0):
        sys.stdout.flush()
        

f.flush()    
sys.stdout.flush()

f.close()

endTotal = timer()
print("time elapsed total: " , endTotal-startTotal, rank)

# comm.Barrier()

# if(rank == 0):
#     tangentFile = folderTangent + 'tangent_{0}_{1}.hd5'
#     tangentFileMerged = folderTangent + 'tangent_{0}.hd5'.format(model)
#     os.system('rm ' + tangentFileMerged)
#     myhd.merge([tangentFile.format(model,i) for i in range(num_ranks)], tangentFileMerged, 
#                 InputLabels = ['id','tangent', 'center'], OutputLabels = ['id','tangent', 'center'], axis = 0, mode = 'w-')
    
#     [os.system('rm ' + tangentFile.format(model,i)) for i in range(num_ranks)]
    
    
    
    # np.savetxt("time_elapsed_reduced_{0}.txt".format(model), np.array([end-start]))
