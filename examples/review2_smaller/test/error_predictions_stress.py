#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:54:51 2022

@author: felipefr
"""


import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np


from deepBND.__init__ import * 
import deepBND.creation_model.training.wrapper_tensorflow as mytf
import deepBND.creation_model.RB.RB_utils as rbut
import tensorflow as tf
# from deepBND.creation_model.training.net_arch import standardNets
from deepBND.creation_model.training.net_arch import NetArch
import deepBND.core.data_manipulation.utils as dman
import deepBND.core.data_manipulation.wrapper_h5py as myhd
from fetricks.fenics.mesh.mesh import Mesh 
import deepBND.core.multiscale.micro_model as mscm
from deepBND.core.multiscale.mesh_RVE import buildRVEmesh
# from deepBND.core.multiscale.micro_model_gen import MicroConstitutiveModelGen # or _new
from deepBND.core.multiscale.micro_model_gen_debug import MicroConstitutiveModelGen # or _new

import dolfin as df # apparently this import should be after the tensorflow stuff

from deepBND.creation_model.prediction.NN_elast_positions_6x6 import NNElast_positions_6x6
import fetricks as ft

standardNets = {'big_A': NetArch([300, 300, 300], 3*['swish'] + ['linear'], 5.0e-4, 0.1, 5*[0.0], 1.0e-8),
                'big_S': NetArch([300, 300, 300], 3*['swish'] + ['linear'], 5.0e-4, 0.1, 5*[0.0], 1.0e-8),            
                'big_tri_A': NetArch([150, 300, 500], 3*['swish'] + ['linear'], 5.0e-4, 0.1, 2*[0.0] + [0.005] + 2*[0.0], 1.0e-8),
                'big_tri_S': NetArch([150, 300, 500], 3*['swish'] + ['linear'], 5.0e-4, 0.1, 2*[0.0] + [0.005] + 2*[0.0], 1.0e-8)}


def getScaledXY(net, Ylabel):
    scalerX, scalerY = dman.importScale(net.files['scaler'], nX, Nrb, scalerType = 'MinMax11')
    Xbar, Ybar = dman.getDatasetsXY(nX, Nrb, net.files['XY'], scalerX, scalerY, Ylabel = Ylabel)[0:2]
    
    return Xbar, Ybar, scalerX, scalerY
    
def loadYlist(net, Ylabel, ids):
    dummy1, Ybar, dummy2, scalerY = getScaledXY(net, Ylabel)
    Y = scalerY.inverse_transform(Ybar[ids])
    return Y 

def predictYlist(net, Ylabel, ids):
    model = net.getModel()   
    model.load_weights(net.files['weights'])
    
    Xbar, dummy1, dummy2, scalerY = getScaledXY(net, Ylabel)
    Yp = scalerY.inverse_transform(model.predict(Xbar[ids])) 

    return Yp

def compute_total_error_ref(ns, loadtype, suffix, bndMeshname, snapshotsname, paramRVEname):
    
    Mref = Mesh(bndMeshname)
    Vref = df.VectorFunctionSpace(Mref,"CG", 2)
    uD = df.Function(Vref)

    contrast = 10.0
    E2 = 1.0
    nu = 0.3
    paramMaterial = [nu,E2*contrast,nu,E2]

    opModel = 'lag'

    # ids = myhd.loadhd5(paramRVEname, "param")[:ns]
    ids = np.arange(ns, dtype = 'int')
    paramRVEdata = myhd.loadhd5(paramRVEname, "param")[ids] 
    Isol = myhd.loadhd5( snapshotsname, 'solutions_%s_%s'%(suffix, loadtype))[ids, :] 
    sigmaTrue = myhd.loadhd5( snapshotsname, 'sigma_%s'%loadtype)[ids, :] 
    
    meshname = "temp.xdmf"
        
    sigmaP = np.zeros((ns,3))
    loadtypes_indexes = {'A': 0, 'S': 2}
    j_voigt = loadtypes_indexes[loadtype]
    
    for i in range(ns):
        buildRVEmesh(paramRVEdata[i,:,:], meshname, 
                      isOrdered = False, size = 'reduced', NxL = 2, NyL = 2, maxOffset = 2, lcar = 2/30)
        
        microModel = mscm.MicroModel(meshname, paramMaterial, opModel)
        
        uD.vector().set_local(Isol[i,:])
        microModel.others['uD'] = uD
        
        microModel.compute()
        
        sigmaP[i,:] = microModel.homogenise([0,1],j_voigt).flatten()[[0,3,2]]    
    

    error = np.mean(np.linalg.norm(sigmaP - sigmaTrue, axis = 1))
    
    return error, sigmaTrue, sigmaP

def compute_total_error_pred(ns, net, loadtype, suffix, bndMeshname, snapshotsname, paramRVEname):
    
    Ylabel = "Y_%s"%loadtype
    
    # Yp = predictYlist(net, Ylabel)
    Yp = loadYlist(net, Ylabel)
    
    Mref = Mesh(bndMeshname)
    Vref = df.VectorFunctionSpace(Mref,"CG", 2)
    uD = df.Function(Vref)

    contrast = 10.0
    E2 = 1.0
    nu = 0.3
    paramMaterial = [nu,E2*contrast,nu,E2]

    opModel = 'dnn'

    dsRef = df.Measure('ds', Mref) 
    
    Wbasis = myhd.loadhd5( net.files['Wbasis'], 'Wbasis_%s'%loadtype)

    # ids = myhd.loadhd5(paramRVEname, "param")[:ns]
    ids = np.arange(ns, dtype = 'int')
    paramRVEdata = myhd.loadhd5(paramRVEname, "param")[ids] 
    sigmaTrue = myhd.loadhd5( snapshotsname, 'sigma_%s'%loadtype)[ids, :] 
    
    meshname = "temp.xdmf"
    
    # Mesh 
    sigmaP = np.zeros((ns,3))
    loadtypes_indexes = {'A': 0, 'S': 2}
    j_voigt = loadtypes_indexes[loadtype]
    
    for i in range(ns):
        buildRVEmesh(paramRVEdata[i,:,:], meshname, 
                      isOrdered = False, size = 'reduced', NxL = 2, NyL = 2, maxOffset = 2, lcar = 2/30)
        
        microModel = mscm.MicroModel(meshname, paramMaterial, opModel)
        
        uD.vector().set_local(Wbasis[:net.nY,:].T@Yp[i,:net.nY])
        microModel.others['uD'] = uD
        
        microModel.compute()
    
        sigmaP[i,:] = microModel.homogenise([0,1],j_voigt).flatten()[[0,3,2]]    
        
    error = np.mean(np.linalg.norm(sigmaP - sigmaTrue, axis = 1))
    
    return error, sigmaTrue, sigmaP


def getTangentTrue(snapshotsname, ids):
    ns = len(ids)
    tangentTrue = np.zeros((ns,3,3))
    
    tangentTrue[:,:, 0] = myhd.loadhd5( snapshotsname, 'sigma_A')[ids, :]
    tangentTrue[:,:, 2] = myhd.loadhd5( snapshotsname, 'sigma_S')[ids, :]
    
    tangentTrue[:,0,1] = tangentTrue[:,1,0]
    tangentTrue[:,2,1] = tangentTrue[:,1,2]
    tangentTrue[:,1,1] = tangentTrue[:,0,0]
    
    return tangentTrue

def compute_total_error_tan(op, ns, nets, suffix, bndMeshname, snapshotsname, paramRVEname):
        
    Mref = Mesh(bndMeshname)
    Vref = df.VectorFunctionSpace(Mref,"CG", 2)
    uD = df.Function(Vref)

    contrast = 10.0
    E2 = 1.0
    nu = 0.3
    paramMaterial = [nu,E2*contrast,nu,E2]

    opModel = 'dnn'
    


    # ids = myhd.loadhd5(paramRVEname, "param")[:ns]
    ids = np.arange(ns, dtype = 'int')
    paramRVEdata = myhd.loadhd5(paramRVEname, "param")[ids] 
    tangentTrue = getTangentTrue( snapshotsname, ids) 
    meshname = "temp.xdmf"
    
    if(op == 'ref'):
        sol_p_A = myhd.loadhd5( snapshotsname, 'solutions_%s_%s'%(suffix, 'A'))[ids, :] 
        sol_p_S = myhd.loadhd5( snapshotsname, 'solutions_%s_%s'%(suffix, 'S'))[ids, :] 
    
    else: 
        Wbasis_A = myhd.loadhd5( nets['A'].files['Wbasis'], 'Wbasis_A')
        Wbasis_S = myhd.loadhd5( nets['S'].files['Wbasis'], 'Wbasis_S')
    
        if(op == 'pred' ): 
            Yp_A = predictYlist(nets['A'], "Y_A", ids)
            Yp_S = predictYlist(nets['S'], "Y_S", ids)

            Yp_A_ = loadYlist(nets['A'], "Y_A", ids)
            Yp_S_ = loadYlist(nets['S'], "Y_S", ids)

            error_A = np.sum((Yp_A - Yp_A_)**2, axis = 1) # all modes along each snapshot
            error_S = np.sum((Yp_S - Yp_S_)**2, axis = 1) # all modes along each snapshot
            print(error_A, error_S)
            
        elif(op == 'pod'): 
            Yp_A = loadYlist(nets['A'], "Y_A")
            Yp_S = loadYlist(nets['S'], "Y_S")
            
            
            
        sol_p_A = [ Wbasis_A[:net.nY,:].T@Yp_A[i,:net.nY]  for i in range(ns)]
        sol_p_S = [ Wbasis_S[:net.nY,:].T@Yp_S[i,:net.nY]  for i in range(ns)]
        
    
    # print("op", op)
    # print(sol_p_A)
    # print(sol_p_S)
    
    # input()
    
    
    # Mesh 
    tangentP = np.zeros((ns,3,3))
    
    for i in range(ns):
        buildRVEmesh(paramRVEdata[i,:,:], meshname, 
                      isOrdered = False, size = 'reduced', NxL = 2, NyL = 2, maxOffset = 2, lcar = 2/30)
        
        microModel = MicroConstitutiveModelGen(meshname, paramMaterial, opModel)
        
        microModel.others['uD'] = uD
        # microModel.others['uD0_'] = Wbasis_A[:net.nY,:].T@Yp_A[i,:net.nY] # it was already picked correctly
        # microModel.others['uD1_'] = np.zeros(Vref.dim()) 
        # microModel.others['uD2_'] = Wbasis_S[:net.nY,:].T@Yp_S[i,:net.nY]

        microModel.others['uD0_'] = sol_p_A[i] 
        microModel.others['uD1_'] = np.zeros(Vref.dim()) 
        microModel.others['uD2_'] = sol_p_S[i] 
        
        Hom = microModel.getHomogenisation()
        tangentP[i,:, : ] = Hom['sigmaL']    
        
    dtan = tangentP - tangentTrue
    error = np.mean(np.linalg.norm(dtan[:,[0,2]].reshape((-1,6)), axis = 1))
    
    return error, tangentTrue, tangentP


def export_vtk_predictions(nets, ns, bndMeshname):
    
    Mref = Mesh(bndMeshname)
    Vref = df.VectorFunctionSpace(Mref,"CG", 2)
    uD = df.Function(Vref)

    ids = np.arange(ns, dtype = 'int')
    
    Yp_A = predictYlist(nets['A'], "Y_A",  ids)
    Yt_A = loadYlist(nets['A'], "Y_A", ids)
    
    Yp_S = predictYlist(nets['S'], "Y_S",  ids)
    Yt_S = loadYlist(nets['S'], "Y_S", ids)
    
    
    Wbasis_A = myhd.loadhd5( nets['A'].files['Wbasis'], 'Wbasis_A')
    Wbasis_S = myhd.loadhd5( nets['S'].files['Wbasis'], 'Wbasis_S')
    
    M = myhd.loadhd5( nets['S'].files['Wbasis'], 'massMat')
    mask = np.sum(np.abs(M), axis = 1).flatten()
    mask[mask.nonzero()] = 1.0
    
    sol_p_A = [ Wbasis_A[:net.nY,:].T@Yp_A[i,:net.nY]  for i in range(ns)]
    sol_p_S = [ Wbasis_S[:net.nY,:].T@Yp_S[i,:net.nY]  for i in range(ns)]
    
    sol_t_A = [ Wbasis_A[:net.nY,:].T@Yt_A[i,:net.nY]  for i in range(ns)]
    sol_t_S = [ Wbasis_S[:net.nY,:].T@Yt_S[i,:net.nY]  for i in range(ns)]
    
    sol_A = myhd.loadhd5( snapshotsname, 'solutions_%s_%s'%(suffix, 'A'))[ids, :] 
    sol_S = myhd.loadhd5( snapshotsname, 'solutions_%s_%s'%(suffix, 'S'))[ids, :] 
    
    
    for i in range(ns):
        for sol, name in zip([sol_p_A, sol_p_S, sol_t_A, sol_t_S, sol_A, sol_S],
                       ['sol_p_A', 'sol_p_S', 'sol_t_A', 'sol_t_S', 'sol_A', 'sol_S']):
            
            uD.vector().set_local(mask*sol[i])
            with df.XDMFFile("%s_%d.xdmf"%(name,i)) as f:
                f.write(uD)
        
   

folder = rootDataPath + '/review2_smaller/'
folderDataset = folder + 'dataset/'
folderPrediction = folder + 'prediction/'
folderTrain = folder + 'training/'
folderBasis = folder + 'dataset/'

nameMeshRefBnd = folderBasis + 'boundaryMesh.xdmf'
snapshotsname = folderDataset + 'snapshots.hd5'
paramRVEname = folderDataset + 'paramRVEdataset.hd5'
suffix = 'translation'
label = 'A' 
ns = 1

archId = 'big'
Nrb = 600
nX = 72

nets = {}

for label in ['A', 'S']: 
    nets[label] = standardNets[archId + '_' +  label] 
    net = nets[label]
    net.nY = Nrb
    net.nX = nX
    net.files['XY'] = folderDataset + 'XY_%s.hd5'%suffix 
    net.files['Wbasis'] = folderDataset + 'Wbasis_%s.hd5'%suffix
    net.files['weights'] = folderTrain + 'models_weights_%s_%s_%d_%s.hdf5'%(archId, label, Nrb, suffix)
    net.files['scaler'] = folderTrain + 'scaler_%s_%s.txt'%(suffix, label)
    net.files['hist'] = folderTrain + 'models_weights_%s_%s_%d_%s_plot_history_val.txt'%(archId, label, Nrb, suffix) 
    
# error = compute_total_error_ref(ns, label, suffix, nameMeshRefBnd, snapshotsname, paramRVEname)
# error_pred = compute_total_error_pred(ns, net, label, suffix, nameMeshRefBnd, snapshotsname, paramRVEname)

# error_tan_ref = compute_total_error_tan('ref', ns, nets, suffix, nameMeshRefBnd, snapshotsname, paramRVEname)
# error_tan_pod = compute_total_error_tan('pod', ns, nets, suffix, nameMeshRefBnd, snapshotsname, paramRVEname)
# error_tan_pred = compute_total_error_tan('pred', ns, nets, suffix, nameMeshRefBnd, snapshotsname, paramRVEname)

# print(error[0])
# print(error_pred[0])
# print(error_tan_ref[0], error_tan_pod[0], error_tan_pred[0])

# print(np.mean(np.linalg.norm((error_tan_pod[2] - error_tan_ref[2])[:,[0,2]].reshape((-1,6)), axis = 1)))
# print(np.mean(np.linalg.norm((error_tan_pred[2] - error_tan_ref[2])[:,[0,2]].reshape((-1,6)), axis = 1)))


# export_vtk_predictions(nets, ns, nameMeshRefBnd)

W0, W, M = rbut.test_zerofiedBasis(folderDataset + 'Wbasis_%s_zerofied.hd5'%suffix, folderDataset + 'Wbasis_%s.hd5'%suffix)