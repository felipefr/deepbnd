import sys, os
import numpy as np
import dolfin as df 
import matplotlib.pyplot as plt
from ufl import nabla_div
from timeit import default_timer as timer
from mpi4py import MPI

from deepBND.__init__ import *
import deepBND.core.data_manipulation.wrapper_h5py as myhd
from deepBND.core.fenics_tools.enriched_mesh import EnrichedMesh 
from deepBND.core.multiscale.micro_model_dnn import MicroConstitutiveModelDNN
from deepBND.core.multiscale.mesh_RVE import buildRVEmesh

# for i in {0..19}; do nohup python computeTangents_serial.py 24 $i 20 > log_ny24_full_per_run$i.py & done

comm = MPI.COMM_WORLD
comm_self = MPI.COMM_SELF

def predictTangents(num, num_runs, modelBnd, namefiles, createMesh, meshSize):
    
    nameMeshRefBnd, paramRVEname, tangentName, BCname, meshMicroName = namefiles
    
    start = timer()
    
    # loading boundary reference mesh
    Mref = EnrichedMesh(nameMeshRefBnd,comm_self)
    Vref = df.VectorFunctionSpace(Mref,"CG", 1)
    
    dxRef = df.Measure('dx', Mref) 
    
    # defining the micro model
    
    paramRVEdata = myhd.loadhd5(paramRVEname, 'param')[run::num_runs]
    ns = len(paramRVEdata) # per rank
    
    if(myhd.checkExistenceDataset(paramRVEname, 'id')):    
        ids = myhd.loadhd5(paramRVEname, 'id')[run::num_runs].astype('int')
        centers = myhd.loadhd5(paramRVEname, 'center')[ids,:]
    else: # dummy
        ids = np.zeros(ns)
        centers = np.zeros((ns,2))
       
    os.system('rm ' + tangentName)
    Iid_tangent_center, f = myhd.zeros_openFile(tangentName, [(ns,), (ns,3,3), (ns,2)],
                                           ['id', 'tangent','center'], mode = 'w')
    
    Iid, Itangent, Icenter = Iid_tangent_center
    
    if(modelBnd == 'dnn'):
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
        meshMicroName_i = meshMicroName.format(int(Iid[i]), meshSize)
        
        if(createMesh):
            buildRVEmesh(paramRVEdata[i,:,:], meshMicroName_i, isOrdinated = False, size = meshSize)
    
        microModel = MicroConstitutiveModelDNN(meshMicroName_i, param, modelBnd) 
        
        if(modelBnd == 'dnn'):
            microModel.others['uD'] = df.Function(Vref) 
            microModel.others['uD0_'] = u0_p[i] # it was already picked correctly
            microModel.others['uD1_'] = u1_p[i] 
            microModel.others['uD2_'] = u2_p[i]
        elif(modelBnd == 'lin'):
            microModel.others['uD'] = df.Function(Vref) 
            microModel.others['uD0_'] = np.zeros(Vref.dim())
            microModel.others['uD1_'] = np.zeros(Vref.dim())
            microModel.others['uD2_'] = np.zeros(Vref.dim())
            
        
        Icenter[i,:] = centers[i,:]
        Itangent[i,:,:] = microModel.getTangent()
        
        if(i%10 == 0):
            f.flush()    
            sys.stdout.flush()
            
    f.close()

if __name__ == '__main__':
    
    if(len(sys.argv)>1):
        run = int(sys.argv[1])
        num_runs = int(sys.argv[2])
    else:
        run = comm.Get_rank()
        num_runs = comm.Get_size()
    
    print('run, num_runs ', run, num_runs)
    
    suffixTangent = 'dnn'
    modelBnd = 'dnn'
    meshSize = 'reduced'
    createMesh = True

    if(modelBnd == 'dnn'):
        modelDNN = '_small_80' # underscore included before
    else:
        modelDNN = ''
               
    folder = rootDataPath + "/deepBND/"
    folderPrediction = folder + 'prediction/'
    folderMesh = folderPrediction + 'meshes/'
    folderDataset = folder + 'dataset/'
    paramRVEname = folderPrediction + 'paramRVEdataset_validation.hd5' 
    nameMeshRefBnd = folderDataset + 'boundaryMesh.xdmf'
    tangentName = folderPrediction + 'tangents_{0}_{1}.hd5'.format(suffixTangent,run)
    BCname = folderPrediction + 'bcs%s.hd5'%modelDNN
    meshMicroName = folderMesh + 'mesh_micro_{0}_{1}.xdmf'

    namefiles = [nameMeshRefBnd, paramRVEname, tangentName, BCname, meshMicroName]
    
    predictTangents(run, num_runs, modelBnd, namefiles, createMesh, meshSize)
    


