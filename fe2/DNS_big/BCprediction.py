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
from tensorflow_for_training import *
import tensorflow as tf
import symmetryLib as syml
from timeit import default_timer as timer
import multiphenics as mp

permY = np.array([2,0,3,1,12,10,8,4,13,5,14,6,15,11,9,7,30,28,26,24,22,16,31,17,32,18,33,19,34,20,35,29,27,25,23,21])

# loading boundary reference mesh
nameMeshRefBnd = '../boundaryMesh.xdmf'
Mref = meut.EnrichedMesh(nameMeshRefBnd)
Vref = df.VectorFunctionSpace(Mref,"CG", 1)
# normal = FacetNormal(Mref)
# volMref = 4.0
    
# loading the DNN model
ellipseData = myhd.loadhd5('./DNS_72/ellipseData_RVEs_volFrac.hd5', 'ellipseData') 

modelDNN = 'big_140'
Nrb = 140
archId = 1
nX = 36
folderBasisShear = '../models/dataset_partially_shear1/'
folderBasisAxial = '../models/dataset_partially_axial1/'
nameWbasisShear = folderBasisShear +  'Wbasis.h5'
nameWbasisAxial = folderBasisAxial +  'Wbasis.h5'
nameScaleXY_shear = folderBasisShear +  '/models/scaler_{0}.txt'.format(Nrb) # chaged
nameScaleXY_axial = folderBasisAxial +  '/models/scaler_{0}.txt'.format(Nrb)

nets = {}
nets['big'] = {'Neurons': [300, 300, 300], 'activations': 3*['swish'] + ['linear'], 'lr': 5.0e-4, 'decay' : 0.1, 'drps' : [0.0] + 3*[0.005] + [0.0], 'reg' : 1.0e-8}
nets['small'] = {'Neurons': [40, 40, 40], 'activations': 3*['swish'] + ['linear'], 'lr': 5.0e-4, 'decay' : 0.1, 'drps' : [0.0] + 3*[0.005] + [0.0], 'reg' : 1.0e-8}
nets['medium'] = {'Neurons': [100, 100, 100], 'activations': 3*['swish'] + ['linear'], 'lr': 5.0e-4, 'decay' : 0.01, 'drps' : [0.0] + 3*[0.005] + [0.0], 'reg' : 1.0e-8}

net = nets[modelDNN.split('_')[0]]

net['nY'] = Nrb
net['nX'] = nX
net['file_weights_shear'] = folderBasisShear + 'models/weights_{0}.hdf5'.format(modelDNN)
net['file_weights_axial'] = folderBasisAxial + 'models/weights_{0}.hdf5'.format(modelDNN)

scalerX_shear, scalerY_shear  = importScale(nameScaleXY_shear, nX, Nrb)
scalerX_axial, scalerY_axial  = importScale(nameScaleXY_axial, nX, Nrb)

Wbasis_shear = myhd.loadhd5(nameWbasisShear, 'Wbasis')
Wbasis_axial = myhd.loadhd5(nameWbasisAxial, 'Wbasis')

X_shear_s = scalerX_shear.transform(ellipseData[:,:,2])
X_axial_s = scalerX_axial.transform(ellipseData[:,:,2])
X_axialY_s = scalerX_axial.transform(ellipseData[:,permY,2])

modelShear = generalModel_dropReg(nX, Nrb, net)   
modelAxial = generalModel_dropReg(nX, Nrb, net)   

modelShear.load_weights(net['file_weights_shear'])
modelAxial.load_weights(net['file_weights_axial'])

Y_p_shear = scalerY_shear.inverse_transform(modelShear.predict(X_shear_s))
Y_p_axial = scalerY_axial.inverse_transform(modelAxial.predict(X_axial_s))
Y_p_axialY = scalerY_axial.inverse_transform(modelAxial.predict(X_axialY_s)) # changed

S_p_shear = Y_p_shear @ Wbasis_shear[:Nrb,:]
S_p_axial = Y_p_axial @ Wbasis_axial[:Nrb,:]
piola_mat = syml.PiolaTransform_matricial('mHalfPi', Vref)
S_p_axialY = Y_p_axialY @ Wbasis_axial[:Nrb,:] @ piola_mat.T #changed

myhd.savehd5('./DNS_72/BCsPrediction_RVEs_volFrac.hd5'.format(modelDNN), 
             [S_p_axial,S_p_axialY, S_p_shear], ['u0','u1','u2'], mode = 'w')
