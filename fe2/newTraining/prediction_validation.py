import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0,'../../utils/')
import matplotlib.pyplot as plt
import numpy as np

import myTensorflow as mytf
from timeit import default_timer as timer

from dolfin import *
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

typeModel = 'shear'
Nrb = 40
archId = 1
nX = 36

def get_mesh(ellipseData, nameMesh, size = 'reduced'):
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
            
    elif(size == 'full'):
        meshGMSH = meut.ellipseMesh2Domains(x0L, y0L, LxL, LyL, NL, ellipseData[:36,:], Lxt, Lyt, lcar, x0 = x0, y0 = y0)
        meshGMSH.setTransfiniteBoundary(NpLxt)
        meshGMSH.setTransfiniteInternalBoundary(NpLxL)   

    meshGMSH.write(nameMesh, opt = 'fenics')
            
    return meut.EnrichedMesh(nameMesh)
    
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
folderBasisShear = './models/dataset_partially_shear1/' #changed
folderBasisAxial = './models/dataset_partially_axial1/'
nameWbasisShear = folderBasisShear +  'Wbasis.h5'
nameWbasisAxial = folderBasisAxial +  'Wbasis.h5'
nameScaleXY_shear = folderBasisShear +  '/models/scaler.txt' # chaged
nameScaleXY_axial = folderBasisAxial +  '/models/scaler.txt'

folderTest = './models/dataset_axial3/'
ellipseData = myhd.loadhd5(folderTest +  'ellipseData.h5', 'ellipseData') 

scalerX_shear, scalerY_shear  = importScale(nameScaleXY_shear, nX, Nrb)
scalerX_axial, scalerY_axial  = importScale(nameScaleXY_axial, nX, Nrb)

Wbasis_shear = myhd.loadhd5(nameWbasisShear, 'Wbasis')
Wbasis_axial = myhd.loadhd5(nameWbasisAxial, 'Wbasis')

X_shear_s = scalerX_shear.transform(ellipseData[:,:,2])
X_axial_s = scalerX_axial.transform(ellipseData[:,:,2])

permY = np.array([2,0,3,1,12,10,8,4,13,5,14,6,15,11,9,7,30,28,26,24,22,16,31,17,32,18,33,19,34,20,35,29,27,25,23,21])
X_axialY_s = scalerX_axial.transform(ellipseData[:,permY,2])

# 
netBig = {'Neurons': [300, 300, 300], 'activations': 3*['swish'] + ['linear'], 'lr': 5.0e-4, 'decay' : 0.1, 'drps' : [0.0] + 3*[0.005] + [0.0], 'reg' : 1.0e-8}
netSmall = {'Neurons': [40, 40, 40], 'activations': 3*['swish'] + ['linear'], 'lr': 5.0e-4, 'decay' : 0.1, 'drps' : [0.0] + 3*[0.005] + [0.0], 'reg' : 1.0e-8}

net = netSmall

net['nY'] = Nrb
net['nX'] = nX
net['file_weights_shear'] = folderBasisShear + 'models/weights_small.hdf5'
net['file_weights_axial'] = folderBasisAxial + 'models/weights_small.hdf5'

# Prediction 
modelShear = generalModel_dropReg(nX, Nrb, net)   
modelAxial = generalModel_dropReg(nX, Nrb, net)   

modelShear.load_weights(net['file_weights_shear'])
modelAxial.load_weights(net['file_weights_axial'])

Y_p_shear = scalerY_shear.inverse_transform(modelShear.predict(X_shear_s))
Y_p_axial = scalerY_axial.inverse_transform(modelAxial.predict(X_axial_s))
Y_p_axialY = scalerY_axial.inverse_transform(modelAxial.predict(X_axialY_s)) 

S_p_shear = Y_p_shear @ Wbasis_shear[:Nrb,:]
S_p_axial = Y_p_axial @ Wbasis_axial[:Nrb,:]

Y_p_axialY = scalerY_axial.inverse_transform(modelAxial.predict(X_axialY_s)) 

Mref = meut.EnrichedMesh(nameMeshRefBnd)
Vref = VectorFunctionSpace(Mref,"CG", 1)
normal = FacetNormal(Mref)
volMref = 4.0

piola_mat = syml.PiolaTransform_matricial('mHalfPi', Vref)
S_p_axialY = Y_p_axialY @ Wbasis_axial[:Nrb,:] @ piola_mat.T 

epsShear = np.zeros((2,2))
epsShear[1,0] = 0.5
epsShear[0,1] = 0.5

epsAxial = np.zeros((2,2))
epsAxial[0,0] = 1.0

epsAxialY = np.zeros((2,2))
epsAxialY[1,1] = 1.0

ns = len(S_p_shear)
ns_max = 200
np.random.seed(1)
randInd = np.arange(ns,dtype='int') 
np.random.shuffle(randInd)
randInd =  randInd[:ns_max]
np.savetxt('randomIndicesTest_small_seed1.txt', randInd)

os.system('rm sigma_prediction_small_eps1.hd5')
fields, f = myhd.zeros_openFile('sigma_prediction_small_eps1.hd5',
                                [(ns_max,3),(ns_max,3),(ns_max,3),(ns_max,3),(ns_max,3),(ns_max,4),(ns_max,4)], 
                                ['sigma_dnn','sigma_mr','sigma_per','sigma_lin','sigma_ref','error','error_rel'])


sigma_dnn, sigma_MR, sigma_per, sigma_lin, sigma_ref, error, error_rel = fields  

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
elif(typeModel == 'mixed'):
    exx = 0.05
    exy = 0.01
    eyy = 0.1
    
    
    
eps = exx*epsAxial + exy*epsShear + eyy*epsAxialY


for i, ii in enumerate(randInd):  
    print("snaptshots i, ii = ", i, ii)
    mesh = get_mesh(ellipseData[ii], 'mesh_temp_small.xdmf', 'reduced')    

    uB = Function(Vref)   
    uB.vector().set_local(exx*S_p_axial[ii,:] + exy*S_p_shear[ii,:] + eyy*S_p_axialY[ii,:]) 
    
    B = -feut.Integral(outer(uB,normal), Mref.ds, (2,2))/volMref
    T = feut.affineTransformationExpression(np.zeros(2),B, Mref) # ignore a, since the basis is already translated
    
    epsB = eps - B
    
    uB.vector().set_local(uB.vector().get_local()[:] + interpolate(T,Vref).vector().get_local()[:])
    
    sigma_dnn[i,:], a1, B2 = get_stress(mesh,epsB,uB,'reduced')        
    sigma_MR[i,:], a1, B2 = get_stress(mesh,epsB,uB,'reduced', 'MR')
    sigma_per[i,:], a1, B2 = get_stress(mesh,epsB,uB,'reduced', 'periodic')
    sigma_lin[i,:], a1, B2 = get_stress(mesh,epsB,uB,'reduced', 'linear')
    
    meshFull = get_mesh(ellipseData[ii], 'mesh_temp_full_small.xdmf', 'full')
    sigma_ref[i,:], a_ref, B_ref = get_stress(meshFull,eps,[None],'full')

    error[i,0] = normStress(sigma_dnn[i,:] - sigma_ref[i,:])
    error[i,1] = normStress(sigma_MR[i,:] - sigma_ref[i,:])
    error[i,2] = normStress(sigma_per[i,:] - sigma_ref[i,:])
    error[i,3] = normStress(sigma_lin[i,:] - sigma_ref[i,:])
    error_rel[i,0] = normStress(sigma_dnn[i,:] - sigma_ref[i,:])/normStress(sigma_ref[i,:])
    error_rel[i,1] = normStress(sigma_MR[i,:] - sigma_ref[i,:])/normStress(sigma_ref[i,:])
    error_rel[i,2] = normStress(sigma_per[i,:] - sigma_ref[i,:])/normStress(sigma_ref[i,:])
    error_rel[i,3] = normStress(sigma_lin[i,:] - sigma_ref[i,:])/normStress(sigma_ref[i,:])
    
    print(B)
    print(B_ref)
    
    print(error_rel[i,:])
    print(sigma_ref[i,:])
    print(sigma_dnn[i,:])
    
    
f.close()