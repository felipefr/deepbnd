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

# Nrb = int(sys.argv[1])
# Nrb = int(input("Nrb="))

Nrb = 140
archId = 1
nX = 36

def get_mesh(ellipseData):
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
    
    meshGMSH = meut.ellipseMesh2(ellipseData[:4,:], x0L, y0L, LxL, LyL, lcar)
    meshGMSH.setTransfiniteBoundary(NpLxL)
        
    meshGMSH.setNameMesh("mesh_temp_{0}.xml".format(Nrb))
    mesh = meshGMSH.getEnrichedMesh()
    
    return mesh
    
def get_stress(mesh,eps, uB):    
    opModel = 'BCdirich_lag'
    
    contrast = 10.0
    E2 = 1.0
    E1 = contrast*E2 # inclusions
    nu1 = 0.3
    nu2 = 0.3
    
    mu1 = elut.eng2mu(nu1,E1)
    lamb1 = elut.eng2lambPlane(nu1,E1)
    mu2 = elut.eng2mu(nu2,E2)
    lamb2 = elut.eng2lambPlane(nu2,E2)
    param = np.array([[lamb1, mu1], [lamb2,mu2]])
        
    sigma, sigmaEps = fmts.getSigma_SigmaEps(param,mesh,eps, op = 'cpp')
    
    # # Solving with Multiphenics
    others = {'method' : 'default', 'polyorder' : 2, 'uD' : uB}
    U = mpms.solveMultiscale(param, mesh, eps, op = opModel, others = others)
    
    sigma_rec = fmts.homogenisation(U[0], mesh, sigma, [0,1], sigmaEps).flatten()[[0,3,2]]  
    
    return sigma_rec





nameMeshRefBnd = 'boundaryMesh.xdmf'
folderBasis = './models/dataset_axial1/'
folder = './models/dataset_axial3/'
nameSnaps = folder + 'snapshots.h5'
nameXY = folder +  'XY.h5'
nameWbasis = folderBasis +  'Wbasis.h5'
nameScaleXY = folderBasis +  'scaler.txt'
nameXYtest = folder + 'XY_Wbasis1.h5'

scalerX, scalerY  = importScale(nameScaleXY, nX, Nrb)
Wbasis = myhd.loadhd5(nameWbasis, 'Wbasis')

X_s, Y_s = syml.getDatasetsXY(nX, Nrb, nameXYtest, scalerX, scalerY)[0:2]

# 
net = {'Neurons': 3*[300], 'activations': 3*['swish'] + ['linear'], 'lr': 5.0e-4, 'decay' : 0.1, 'drps' : [0.0] + 3*[0.005] + [0.0], 'reg' : 1.0e-8}

net['nY'] = Nrb
net['nX'] = nX
net['file_weights'] = folderBasis + 'models/weights_ny{0}_arch{1}.hdf5'.format(Nrb,archId)


# Prediction 
model = generalModel_dropReg(nX, Nrb, net)   

model.load_weights(net['file_weights'])

Y_p_s = model.predict(X_s)
Y_p = scalerY.inverse_transform(Y_p_s)

S_p = Y_p @ Wbasis[:Nrb,:]

Isol = myhd.loadhd5(nameSnaps,'solutions_trans')
Bsnaps = myhd.loadhd5(nameSnaps,'B')

Mref = meut.EnrichedMesh(nameMeshRefBnd)
Vref = VectorFunctionSpace(Mref,"CG", 1)

dxRef = Measure('dx', Mref) 
dsRef = Measure('ds', Mref) 

ellipseData = myhd.loadhd5(folder +  'ellipseData.h5', 'ellipseData') # unique, not partitioned
epsAxial = np.zeros((2,2))
epsAxial[0,0] = 1.0

ns = len(S_p)
ns_max = 100
np.random.seed(1)
randInd = np.arange(ns,dtype='int') 
np.random.shuffle(randInd)
randInd =  randInd[:ns_max]
np.savetxt('randomIndicesTest_axial.txt', randInd)

os.system('rm sigma_prediction_ny{0}.hd5'.format(Nrb))
fields, f = myhd.zeros_openFile('sigma_prediction_ny{0}.hd5'.format(Nrb),
                                [(ns_max,3),(ns_max,3),(ns_max,1)], ['sigma','sigma_ref','error'])
sigma, sigma_ref, error = fields  

normStress = lambda s : np.sqrt(s[0]**2 + s[1]**2 + 2*s[2]**2)

for i, ii in enumerate(randInd):  
    print("snaptshots i, ii = ", i, ii)
    mesh = get_mesh(ellipseData[ii])
    eps = epsAxial - Bsnaps[ii,:,:]

    uB = Function(Vref)
    uB_ref = Function(Vref)
    
    uB.vector().set_local(S_p[ii,:])
    uB_ref.vector().set_local(Isol[ii,:])

    sigma[i,:] = get_stress(mesh,eps,uB)
    sigma_ref[i,:] = get_stress(mesh,eps,uB_ref)
    error[i,0] = normStress(sigma[i,:] - sigma_ref[i,:])
    
    print(sigma[i,:], sigma_ref[i,:],  error[i,0])

f.close()