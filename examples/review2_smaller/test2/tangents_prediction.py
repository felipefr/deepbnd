#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 14:25:42 2022

@author: felipe
"""

import sys, os
import numpy as np
import dolfin as df 
import matplotlib.pyplot as plt
from ufl import nabla_div
from timeit import default_timer as timer

from deepBND.__init__ import *
import fetricks.data_manipulation.wrapper_h5py as myhd
from fetricks.fenics.mesh.mesh import Mesh 
# from deepBND.core.multiscale.micro_model_gen import MicroConstitutiveModelGen
from deepBND.core.multiscale.micro_model_gen_new import MicroConstitutiveModelGen
# from deepBND.core.multiscale.micro_model_dnn import MicroConstitutiveModelDNN
from deepBND.core.multiscale.mesh_RVE import buildRVEmesh

# split BC prediction and paramRVEname



# for k in range(nb_tries):
#     print("Solving not completed simulation: {0}-th chance".format(k))
#     not_completed = []

#     for i in indexes_to_solve:
#         print("Solving snapshot", int(ids[i]), i)
#         try: 
            
#             if(createMesh):
#                 buildRVEmesh(paramRVEdata[i,:,:], meshname, 
#                               isOrdered = False, size = 'full', NxL = 2, NyL = 2, maxOffset = 2, lcar = 2/30)
            
#             solve_snapshot(i, meshname, paramMaterial, opModel, snapshots, usol)    
            
#             ids[i] = ids_param[i]
            
#         except:
#             print("Failed solving snapshot", int(ids[i]), i)
#             ids[i] = -1
#             not_completed.append(i)
#         finally:
#             fsnaps.flush()
        
#     indexes_to_solve = np.array(not_completed)
#     print(indexes_to_solve)
#     if(len(indexes_to_solve) == 0):
#         break
    

def predictTangents(ns, modelBnd, namefiles, createMesh, meshSize):
    
    nameMeshRefBnd, paramRVEname, tangentName, BCname, meshMicroName = namefiles
    
    # loading boundary reference mesh
    Mref = Mesh(nameMeshRefBnd)
    Vref = df.VectorFunctionSpace(Mref,"CG", 2)
    
    dxRef = df.Measure('dx', Mref) 
    
    # defining the micro model
    ids = myhd.loadhd5(paramRVEname, 'ids')[:ns].flatten().astype('int')
    paramRVEdata = myhd.loadhd5(paramRVEname, 'param')[:ns]
    
    os.system('rm ' + tangentName)
    Iid_tangent_eps, f = myhd.zeros_openFile(tangentName, [(ns,), (ns,3,3), (ns,3,3), (ns,3,3), (ns,3,3)],
                                           ['id', 'tangent', 'tangentT', 'eps', 'epsT'], mode = 'w')
    
    Iid, Itangent, ItangentT, Ieps, IepsT = Iid_tangent_eps
    
    if(modelBnd == 'dnn'):
        u0_p = myhd.loadhd5(BCname, 'u0')
        u1_p = myhd.loadhd5(BCname, 'u1')
        u2_p = myhd.loadhd5(BCname, 'u2')
    
    
    for i in range(ns):
    
        Iid[i] = ids[i]
        
        contrast = 10.0
        E2 = 1.0
        nu = 0.3
        param = [nu,E2*contrast,nu,E2]
        print(paramRVEname, i, ids[i])
        meshMicroName_i = meshMicroName.format(int(Iid[i]), meshSize)
    
        start = timer()
        
        buildRVEmesh(paramRVEdata[i,:,:], meshMicroName_i, isOrdered = False, size = meshSize, 
                     NxL = 2, NyL = 2, maxOffset = 2, lcar = 2/30)
        
        end = timer()
        print("time expended in meshing ", end - start)
    
        # microModel = MicroCo=nstitutiveModelDNN(meshMicroName_i, param, modelBnd) 
        microModel = MicroConstitutiveModelGen(meshMicroName_i, param, modelBnd)
        
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
            
    
        Hom = microModel.getHomogenisation()
        Itangent[i,:,:] = Hom['tangentL']
        ItangentT[i,:,:] = Hom['tangent']
        Ieps[i,:,:] = Hom['epsL']
        IepsT[i,:,:] = Hom['eps']
        
        if(i%10 == 0):
            f.flush()    
            sys.stdout.flush()
            
    f.close()



def predictTangents_robust(ns, modelBnd, namefiles, createMesh, meshSize):
    
    nameMeshRefBnd, paramRVEname, tangentName, BCname, meshMicroName = namefiles
    
    # loading boundary reference mesh
    Mref = Mesh(nameMeshRefBnd)
    Vref = df.VectorFunctionSpace(Mref,"CG", 2)

    # defining the micro model
    ids = myhd.loadhd5(paramRVEname, 'ids')[:ns].flatten().astype('int')
    paramRVEdata = myhd.loadhd5(paramRVEname, 'param')[:ns]
    
    os.system('rm ' + tangentName)
    Iid_tangent_eps, f = myhd.zeros_openFile(tangentName, [(ns,), (ns,3,3), (ns,3,3), (ns,3,3), (ns,3,3)],
                                           ['id', 'tangent', 'tangentT', 'eps', 'epsT'], mode = 'w')
    
    Iid, Itangent, ItangentT, Ieps, IepsT = Iid_tangent_eps
    
    if(modelBnd == 'dnn'):
        u0_p = myhd.loadhd5(BCname, 'u0')
        u1_p = myhd.loadhd5(BCname, 'u1')
        u2_p = myhd.loadhd5(BCname, 'u2')
        
    indexes_to_solve = np.arange(ns).astype('int')
    nb_tries = 8
    
    for k in range(nb_tries):

        print("Solving not completed simulation: {0}-th chance".format(k))
        not_completed = []

        for i in indexes_to_solve:
            
            try:
                Iid[i] = ids[i]
                
                contrast = 10.0
                E2 = 1.0
                nu = 0.3
                param = [nu,E2*contrast,nu,E2]
                print(paramRVEname, i, ids[i])
                meshMicroName_i = meshMicroName.format(int(Iid[i]), meshSize)
            
                start = timer()
                
                buildRVEmesh(paramRVEdata[i,:,:], meshMicroName_i, isOrdered = False, size = meshSize, 
                             NxL = 2, NyL = 2, maxOffset = 2, lcar = 2/30)
                
                end = timer()
                print("time expended in meshing ", end - start)
            
                # microModel = MicroCo=nstitutiveModelDNN(meshMicroName_i, param, modelBnd) 
                microModel = MicroConstitutiveModelGen(meshMicroName_i, param, modelBnd)
                
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
                    
            
                Hom = microModel.getHomogenisation()
                Itangent[i,:,:] = Hom['tangentL']
                ItangentT[i,:,:] = Hom['tangent']
                Ieps[i,:,:] = Hom['epsL']
                IepsT[i,:,:] = Hom['eps']

            except:
                print("Failed solving snapshot", int(ids[i]), i)
                Iid[i] = -1; ids[i] = -1
                not_completed.append(i)
            finally:
                f.flush()
                
        indexes_to_solve = np.array(not_completed)
        print(indexes_to_solve)
        if(len(indexes_to_solve) == 0):
            break
        
                
    f.close()

    return indexes_to_solve

# for i in {0..31}; do nohup python tangents_predictions_simplified.py $i > log_$i.txt & done
if __name__ == '__main__':
    
    run = 0
    
    suffixTangent = 'full'
    modelBnd = 'per'
    meshSize = 'full'
    createMesh = True
    suffix = "translation"
    ns = 5

    if(modelBnd == 'dnn'):
        modelDNN = 'big' # underscore included before
    else:
        modelDNN = ''

    folder = rootDataPath + "/review2_smaller/"  
    folderPrediction = folder + 'prediction_test2/'
    # folderMesh = folder + '/prediction/meshes/' # reusing meshes of the other case
    folderMesh = folder + 'prediction_test2/meshes/' 
    paramRVEname = folderPrediction + 'paramRVEdataset_test.hd5' 
    nameMeshRefBnd = folderPrediction + 'boundaryMesh.xdmf'
    tangentName = folderPrediction + 'tangents_{0}.hd5'.format(suffixTangent)
    BCname = folderPrediction + 'bcs_{0}_big_600_test_deltaChanged.hd5'.format(suffix) 
    meshMicroName = folderMesh + 'mesh_micro_{0}_{1}.xdmf'

    namefiles = [nameMeshRefBnd, paramRVEname, tangentName, BCname, meshMicroName]
    
    failed_ids = predictTangents_robust(ns, modelBnd, namefiles, createMesh, meshSize)
    
    print(failed_ids)
