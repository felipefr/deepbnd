import sys, os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import dolfin as df
import matplotlib.pyplot as plt
from ufl import nabla_div
import numpy as np

from deepBND.__init__ import *
import deepBND.creation_model.training.wrapper_tensorflow as mytf
from deepBND.creation_model.training.net_arch import NetArch
import deepBND.core.data_manipulation.utils as dman
import deepBND.core.data_manipulation.wrapper_h5py as myhd
from deepBND.core.fenics_tools.enriched_mesh import EnrichedMesh 
import deepBND.core.fenics_tools.misc as feut


def predictBCs(namefiles, net):
    
    nameMeshRefBnd, nameWbasis, paramRVEname, nameScaleXY_shear, nameScaleXY_axial = namefiles
    
    ## it may be changed (permY)
    # the permY below is only valid for the ordenated radius (inside to outsid)
    # permY = np.array([2,0,3,1,12,10,8,4,13,5,14,6,15,11,9,7,30,28,26,24,22,16,31,17,32,18,33,19,34,20,35,29,27,25,23,21])
    # the permY below is only valid for the radius ordenated by rows and columns (below to top {left to right})
    permY = np.array([[(5-j)*6 + i for j in range(6)] for i in range(6)]).flatten() # note that (i,j) -> (Nx-j-1,i)
    
    
    # loading boundary reference mesh
    Mref = meut.EnrichedMesh(nameMeshRefBnd)
    Vref = df.VectorFunctionSpace(Mref,"CG", 1)
    # normal = FacetNormal(Mref)
    # volMref = 4.0
    
    # loading the DNN model
    paramRVEdata = myhd.loadhd5(paramRVEname, 'param')
    
    scalerX_shear, scalerY_shear  = importScale(nameScaleXY_shear, nX, Nrb)
    scalerX_axial, scalerY_axial  = importScale(nameScaleXY_axial, nX, Nrb)
    
    Wbasis_shear, Wbasis_axial = myhd.loadhd5(nameWbasis, ['Wbasis_S','Wbasis_A'])
    
    X_shear_s = scalerX_shear.transform(paramRVEdata[:,:,2])
    X_axial_s = scalerX_axial.transform(paramRVEdata[:,:,2])
    X_axialY_s = scalerX_axial.transform(paramRVEdata[:,permY,2]) ### permY performs a counterclockwise rotation
    
    modelShear = generalModel_dropReg(nX, Nrb, net)
    modelAxial = generalModel_dropReg(nX, Nrb, net)
    
    modelShear.load_weights(net['file_weights_shear'])
    modelAxial.load_weights(net['file_weights_axial'])
    
    Y_p_shear = scalerY_shear.inverse_transform(modelShear.predict(X_shear_s))
    Y_p_axial = scalerY_axial.inverse_transform(modelAxial.predict(X_axial_s))
    Y_p_axialY = scalerY_axial.inverse_transform(modelAxial.predict(X_axialY_s))
    
    S_p_shear = Y_p_shear @ Wbasis_shear[:Nrb,:]
    S_p_axial = Y_p_axial @ Wbasis_axial[:Nrb,:]
    
    theta = 3*np.pi/2.0 # 'minus HalfPi'
    piola_mat = feut.PiolaTransform_rotation_matricial(theta, Vref)
    S_p_axialY = Y_p_axialY @ Wbasis_axial[:Nrb,:] @ piola_mat.T #changed
    
    myhd.savehd5(bcs_namefile, [S_p_axial,S_p_axialY, S_p_shear], ['u0','u1','u2'], mode = 'w')

if __name__ == '__main__':
    Ny_DNS = 72
    
    folder = rootData + "/new_fe2/"
    folderTrain = folder + 'training/'
    folderBasis = folder + 'dataset/'
    folderDNS = folder + "DNS/DNS_%d_2/"%Ny_DNS
    nameMeshRefBnd = folderBasis + 'boundaryMesh.xdmf'
    
    modelDNN = 'big'
    Nrb = 140
    archId = 1
    nX = 36
    nameWbasis = folderBasis +  'Wbasis.h5'
    nameScaleXY_shear = folderTrain +  'scalers/scaler_S_{0}.txt'.format(Nrb) # changed
    nameScaleXY_axial = folderTrain +  'scalers/scaler_A_{0}.txt'.format(Nrb) # changed
    
    nets = {}
    nets['big'] = {'Neurons': [300, 300, 300], 'activations': 3*['swish'] + ['linear'], 'lr': 5.0e-4, 'decay' : 0.1, 'drps' : [0.0] + 3*[0.005] + [0.0], 'reg' : 1.0e-8}
    nets['small'] = {'Neurons': [40, 40, 40], 'activations': 3*['swish'] + ['linear'], 'lr': 5.0e-4, 'decay' : 0.1, 'drps' : [0.0] + 3*[0.005] + [0.0], 'reg' : 1.0e-8}
    nets['medium'] = {'Neurons': [100, 100, 100], 'activations': 3*['swish'] + ['linear'], 'lr': 5.0e-4, 'decay' : 0.01, 'drps' : [0.0] + 3*[0.005] + [0.0], 'reg' : 1.0e-8}
    
    net = nets[modelDNN.split('_')[0]]
    
    net['nY'] = Nrb
    net['nX'] = nX
    net['file_weights_shear'] = folderTrain + 'models/weights_{0}_S_{1}.hdf5'.format(modelDNN,Nrb)
    net['file_weights_axial'] = folderTrain + 'models/weights_{0}_A_{1}.hdf5'.format(modelDNN,Nrb)
    
    paramRVEname = folderDNS + 'param_RVEs_from_DNS.hd5'
    bcs_namefile = folder + 'DNS/DNS_{2}_2/BCsPrediction_RVEs_{0}_{1}.hd5'.format(modelDNN,Nrb,Ny_DNS)

    namefiles = [nameMeshRefBnd, nameWbasis, paramRVEname, nameScaleXY_shear, nameScaleXY_axial]
    
    predictBCs(namefiles, net)


                                                                               