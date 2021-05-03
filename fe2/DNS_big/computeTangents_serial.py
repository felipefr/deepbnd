import sys, os
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

# for i in {0..19}; do nohup python computeTangents_serial.py 24 $i 20 > log_ny24_full_per_run$i.py & done

Ny = int(sys.argv[1])
run = int(sys.argv[2])
num_runs = int(sys.argv[3])
read_indexes = int(sys.argv[4])

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

if(read_indexes>0):
    ids = np.loadtxt(folderTangent + "other_ids.txt").astype('int')
else:
    ids = np.arange(size_ids).astype('int')

ids = ids[run::num_runs]
Centers = myhd.loadhd5(ellipseDataName, 'center')[ids,:]

ns = len(Centers) # per rank

tangentFile = folderTangent + 'tangent_{0}_{1}.hd5'.format(model,run)
os.system('rm ' + tangentFile)
Iid_tangent_center, f = myhd.zeros_openFile(tangentFile, [(ns,), (ns,3,3), (ns,2)],
                                       ['id', 'tangent','center'], mode = 'w')

Iid, Itangent, Icenter = Iid_tangent_center

if(model == 'dnn'):
    u0_p = myhd.loadhd5(BCname, 'u0')[run::num_runs,:]
    u1_p = myhd.loadhd5(BCname, 'u1')[run::num_runs,:]
    u2_p = myhd.loadhd5(BCname, 'u2')[run::num_runs,:]

for i in range(ns):
    Iid[i] = ids[i]
    
    contrast = 10.0
    E2 = 1.0
    nu = 0.3
    param = [nu,E2*contrast,nu,E2]
    print(run, i, ids[i])
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
    Itangent[i,:,:] = microModel.getTangent()
    
    if(i%10 == 0):
        f.flush()    
        sys.stdout.flush()
        
f.close()

