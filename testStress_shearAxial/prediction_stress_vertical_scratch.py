import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0,'../utils/')
import matplotlib.pyplot as plt
import numpy as np

import myTensorflow as mytf
from timeit import default_timer as timer

from dolfin import *
import h5py
import pickle
# import Generator as gene
import myHDF5 
import dataManipMisc as dman 
from sklearn.preprocessing import MinMaxScaler
import miscPosprocessingTools as mpt
import myHDF5 as myhd
# import symmetryLib as syml
import tensorflow as tf
import multiphenicsMultiscale as mpms
import elasticity_utils as elut

from tensorflow_for_training import *
import meshUtils as meut
import fenicsMultiscale as fmts
import fenicsUtils as feut
import symmetryLib as syml
import copy

# Nrb = int(sys.argv[1])
# Nrb = int(input("Nrb="))

typeModel = 'axial'
Nrb = 140
archId = 1
nX = 36

def get_mesh(ellipseData, size = 'reduced'):
    maxOffset = 2
    
    H = 1.0 # size of each square
    NxL = NyL = 2
    NL = NxL*NyL
    x0L = y0L = -H 
    LxL = LyL = 2*H
    lcar = (2/30)*H
    Nx = (NxL+2*maxOffset)
    Ny = (NyL+2*maxOffset)
    Lxt = Nx*H
    Lyt = Ny*H
    NpLxt = int(Lxt/lcar) + 1
    NpLxL = int(LxL/lcar) + 1
    print("NpLxL=", NpLxL) 
    x0 = -Lxt/2.0
    y0 = -Lyt/2.0
    r0 = 0.2*H
    r1 = 0.4*H
    Vfrac = 0.282743
    rm = H*np.sqrt(Vfrac/np.pi)
    
    if(size == 'reduced'):
        meshGMSH = meut.ellipseMesh2(ellipseData[:4,:], x0L, y0L, LxL, LyL, lcar)
        meshGMSH.setTransfiniteBoundary(NpLxL)
            
        meshGMSH.setNameMesh("mesh_temp_{0}_{1}.xml".format(Nrb,typeModel))
        mesh = meshGMSH.getEnrichedMesh()
    elif(size == 'full'):
        meshGMSH = meut.ellipseMesh2Domains(x0L, y0L, LxL, LyL, NL, ellipseData[:36,:], Lxt, Lyt, lcar, x0 = x0, y0 = y0)
        meshGMSH.setTransfiniteBoundary(NpLxt)
        meshGMSH.setTransfiniteInternalBoundary(NpLxL)   
            
        meshGMSH.setNameMesh("mesh_temp_{0}_{1}_full.xml".format(Nrb,typeModel))
        mesh = meshGMSH.getEnrichedMesh()
            
    return mesh
    
def get_stress(mesh,eps, uB, size = 'reduced', model = 'BCdirich_lag'):    
    if(size == 'reduced'):
        if(model == 'linear'):
            uB.vector().set_local(np.zeros(uB.function_space().dim()))
            model = 'BCdirich_lag'
            
        opModel = model
        others = {'method' : 'default', 'polyorder' : 2, 'uD' : uB, 'per': [-1.0, 1.0, -1.0, 1.0]}
    elif(size == 'full'):
        opModel = 'periodic'
        others = {'method' : 'default', 'polyorder' : 2, 'per': [-3.0, 3.0, -3.0, 3.0]}
    
    contrast = 10.0
    E2 = 1.0
    E1 = contrast*E2 # inclusions
    nu1 = 0.3
    nu2 = 0.3
    
    mu1 = elut.eng2mu(nu1,E1)
    lamb1 = elut.eng2lambPlane(nu1,E1)
    mu2 = elut.eng2mu(nu2,E2)
    lamb2 = elut.eng2lambPlane(nu2,E2)
    param = np.array([[lamb1, mu1], [lamb2,mu2],[lamb1,mu1], [lamb2,mu2]])
        
    sigma, sigmaEps = fmts.getSigma_SigmaEps(param,mesh,eps, op = 'cpp')
    
    # # Solving with Multiphenics

    U = mpms.solveMultiscale(param, mesh, eps, op = opModel, others = others)
    
    sigma_rec = fmts.homogenisation(U[0], mesh, sigma, [0,1], sigmaEps).flatten()[[0,3,2]]
    
    
    T, a, B = feut.getAffineTransformationLocal(U[0],mesh,[0,1], justTranslation = False) 
    
    return sigma_rec, a, B


nameMeshRefBnd = 'boundaryMesh.xdmf'
folderBasisShear = './models/dataset_shear1/'
folderBasisAxial = './models/dataset_axial1/'
nameWbasisShear = folderBasisShear +  'Wbasis.h5'
nameWbasisAxial = folderBasisAxial +  'Wbasis.h5'
nameScaleXY_shear = folderBasisShear +  'scaler.txt'
nameScaleXY_axial = folderBasisAxial +  'scaler.txt'

folder = './models/dataset_axial3/'
ellipseData = myhd.loadhd5(folder +  'ellipseData.h5', 'ellipseData') 

scalerX_shear, scalerY_shear  = importScale(nameScaleXY_shear, nX, Nrb)
scalerX_axial, scalerY_axial  = importScale(nameScaleXY_axial, nX, Nrb)

Wbasis_shear = myhd.loadhd5(nameWbasisShear, 'Wbasis')
Wbasis_axial = myhd.loadhd5(nameWbasisAxial, 'Wbasis')

X_shear_s = scalerX_shear.transform(ellipseData[:,:,2])
X_axial_s = scalerX_axial.transform(ellipseData[:,:,2])

# permY = syml.getPermutation('halfPi')
# permY = np.array([2,0,3,1,12,10,8,4,13,5,14,6,15,11,9,7,30,28,26,24,22,16,31,17,32,18,33,19,34,20,35,29,27,25,23,21])
permY = np.array([1,3,0,2,12,10,8,4,13,5,14,6,15,11,9,7,21,23,25,27,29,35,20,34,19,33,18,32,17,31,16,22,24,26,28,30])
X_axialY_s = scalerX_axial.transform(ellipseData[:,permY,2])

# 
net = {'Neurons': 3*[300], 'activations': 3*['swish'] + ['linear'], 'lr': 5.0e-4, 'decay' : 0.1, 'drps' : [0.0] + 3*[0.005] + [0.0], 'reg' : 1.0e-8}

net['nY'] = Nrb
net['nX'] = nX
net['file_weights_shear'] = folderBasisShear + 'models/weights_ny{0}_arch{1}.hdf5'.format(Nrb,archId)
net['file_weights_axial'] = folderBasisAxial + 'models/weights_ny{0}_arch{1}.hdf5'.format(Nrb,archId)

# Prediction 
modelShear = generalModel_dropReg(nX, Nrb, net)   
modelAxial = generalModel_dropReg(nX, Nrb, net)   

modelShear.load_weights(net['file_weights_shear'])
modelAxial.load_weights(net['file_weights_axial'])

Y_p_shear = scalerY_shear.inverse_transform(modelShear.predict(X_shear_s))
Y_p_axial = scalerY_axial.inverse_transform(modelAxial.predict(X_axial_s))
Y_p_axialY = scalerY_axial.inverse_transform(modelAxial.predict(X_axialY_s)) # changed

S_p_shear = Y_p_shear @ Wbasis_shear[:Nrb,:]
S_p_axial = Y_p_axial @ Wbasis_axial[:Nrb,:]

Mref = meut.EnrichedMesh(nameMeshRefBnd)
Vref = VectorFunctionSpace(Mref,"CG", 1)

dxRef = Measure('dx', Mref) 
dsRef = Measure('ds', Mref) 

# piola = syml.PiolaTransform('mHalfPi', Vref)
piola_mat = syml.PiolaTransform_matricial('halfPi', Vref)
# S_p_axialY = Y_p_axialY @ Wbasis_axial[:Nrb,:] @ piola_mat.T #changed
S_p_axialY = Y_p_axialY @ Wbasis_axial[:Nrb,:] @ piola_mat.T #changed

epsShear = np.zeros((2,2))
epsShear[1,0] = 0.5
epsShear[0,1] = 0.5

epsAxial = np.zeros((2,2))
epsAxial[0,0] = 1.0

epsAxialY = np.zeros((2,2))
epsAxialY[1,1] = 1.0

ns = len(S_p_shear)
ns_max = 1
np.random.seed(1)
randInd = np.arange(ns,dtype='int') 
np.random.shuffle(randInd)
randInd =  randInd[:ns_max]
# np.savetxt('randomIndicesTest_10_seed1.txt', randInd)

os.system('rm sigma_prediction_ny{0}_{1}_1.hd5'.format(Nrb,typeModel))
fields, f = myhd.zeros_openFile('sigma_prediction_ny{0}_{1}_1.hd5'.format(Nrb,typeModel),
                                [(ns_max,3),(ns_max,3),(ns_max,3),(ns_max,3),(ns_max,3),(ns_max,3)], 
                                ['sigma','sigma_per','sigma_lin','sigma_ref','error','error_rel'])

sigma, sigma_per, sigma_lin, sigma_ref, error, error_rel = fields  

normStress = lambda s : np.sqrt(s[0]**2 + s[1]**2 + 2*s[2]**2)

if(typeModel == 'shear'):
    exx = 0.0
    exy = 1.0
    eyy = 0.0
elif(typeModel == 'axial'):
    exx = 1.0
    exy = 0.0
    eyy = 0.0
elif(typeModel == 'axialY'):
    exx = 0.0
    exy = 0.0
    eyy = 1.0
    
    
eps = exx*epsAxial + exy*epsShear + eyy*epsAxialY


for i, ii in enumerate(randInd):  
    print("snaptshots i, ii = ", i, ii)
    mesh = get_mesh(ellipseData[ii], 'reduced')
    ellipseDataRot = copy.deepcopy(ellipseData[ii])
    ellipseDataRot[:,2] = ellipseDataRot[permY,2]
    
    meshRot = get_mesh(ellipseDataRot, 'reduced')
    
    meshFull = get_mesh(ellipseData[ii], 'full')
    
    sigma_ref[i,:], a, B = get_stress(meshFull,eps,[0],'full')
    
    # epsB = eps - B
    
    nref = FacetNormal(mesh)

    uB = Function(Vref)   
    uB.vector().set_local(S_p_axial[ii,:])
    print(feut.Integral(outer(uB,nref), ds, (2,2)))
    
    epsB = eps - B + feut.Integral(outer(uB,nref), ds, (2,2))
    # uB.vector().set_local(exx*S_p_axial[ii,:] + exy*S_p_shear[ii,:] + eyy*S_p_axialY[ii,:]) 
    # uB = piola(uB)
    
    # sigma[i,:], a1, B2 = get_stress(mesh,epsB,uB,'reduced', 'linear')
    sigma[i,:], a1, B2 = get_stress(mesh,epsB,uB,'reduced')
    
    theta = -np.pi/2.0
    R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    uB.vector().set_local(S_p_axialY[ii,:])
    mesh.coordinates()[:] = mesh.coordinates()[:]@R
    
    # sigma_per[i,:], a1, B2 = get_stress(mesh,R.T@epsB@R,uB,'reduced', 'linear')
    sigma_per[i,:], a1, B2 = get_stress(meshRot,R.T@epsB@R,uB,'reduced')
    
    
    # sigma_per[i,:], a1, B2 = get_stress(mesh,epsB,uB,'reduced', 'periodic')
    # sigma_lin[i,:], a1, B2 = get_stress(mesh,epsB,uB,'reduced', 'linear')

    # error[i,0] = normStress(sigma[i,:] - sigma_ref[i,:])
    # error[i,1] = normStress(sigma[i,:] - sigma_per[i,:])
    # error[i,2] = normStress(sigma[i,:] - sigma_lin[i,:])
    # error_rel[i,0] = normStress(sigma[i,:] - sigma_ref[i,:])/normStress(sigma_ref[i,:])
    # error_rel[i,1] = normStress(sigma[i,:] - sigma_per[i,:])/normStress(sigma_ref[i,:])
    # error_rel[i,2] = normStress(sigma[i,:] - sigma_lin[i,:])/normStress(sigma_ref[i,:])
    
    # print(normStress(sigma[i,:]-sigma_ref[i,:]))
    # print(normStress(sigma_per[i,:]-sigma_ref[i,:]))
    # print(normStress(sigma_lin[i,:]-sigma_ref[i,:]))
    print(sigma[i,:])
    # print(sigma_ref[i,:])
    print(sigma_per[i,:])
    
f.close()

print(permY)