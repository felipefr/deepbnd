import sys, os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import dolfin as df
import matplotlib.pyplot as plt
from ufl import nabla_div
import numpy as np

from deepBND.__init__ import *
import deepBND.creation_model.training.wrapper_tensorflow as mytf
from deepBND.creation_model.training.net_arch import standardNets
import deepBND.core.data_manipulation.utils as dman
import deepBND.core.data_manipulation.wrapper_h5py as myhd
from deepBND.core.fenics_tools.enriched_mesh import EnrichedMesh 
import deepBND.core.elasticity.fenics_utils as feut


def predictBCs(namefiles, net):
    
    nameMeshRefBnd, nameWbasis, paramRVEname, nameScaleXY_shear, nameScaleXY_axial = namefiles
    
    ## it may be changed (permY)
    # the permY below is only valid for the ordenated radius (inside to outsid)
    # permY = np.array([2,0,3,1,12,10,8,4,13,5,14,6,15,11,9,7,30,28,26,24,22,16,31,17,32,18,33,19,34,20,35,29,27,25,23,21])
    # the permY below is only valid for the radius ordenated by rows and columns (below to top {left to right})
    permY = np.array([[(5-j)*6 + i for j in range(6)] for i in range(6)]).flatten() # note that (i,j) -> (Nx-j-1,i)
    
    
    # loading boundary reference mesh
    Mref = EnrichedMesh(nameMeshRefBnd)
    Vref = df.VectorFunctionSpace(Mref,"CG", 1)
    # normal = FacetNormal(Mref)
    # volMref = 4.0
    
    # loading the DNN model
    paramRVEdata = myhd.loadhd5(paramRVEname, 'param')
    
    scalerX_shear, scalerY_shear  = dman.importScale(nameScaleXY_shear, nX, Nrb)
    scalerX_axial, scalerY_axial  = dman.importScale(nameScaleXY_axial, nX, Nrb)
    
    Wbasis_shear, Wbasis_axial = myhd.loadhd5(nameWbasis, ['Wbasis_S','Wbasis_A'])
    
    X_shear_s = scalerX_shear.transform(paramRVEdata[:,:,2])
    X_axial_s = scalerX_axial.transform(paramRVEdata[:,:,2])
    X_axialY_s = scalerX_axial.transform(paramRVEdata[:,permY,2]) ### permY performs a counterclockwise rotation
    
    modelShear = net.getModel(nX, Nrb)
    modelAxial = net.getModel(nX, Nrb)
    
    modelShear.load_weights(net.param['file_weights_shear'])
    modelAxial.load_weights(net.param['file_weights_axial'])
    
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
    
    archId = 'small'
    Nrb = 80
    nX = 36
    
    folder = rootDataPath + "/deepBND/"
    folderTrain = folder + 'training/'
    folderBasis = folder + 'dataset/'
    folderPrediction = folder + "prediction/"
    nameMeshRefBnd = folderBasis + 'boundaryMesh.xdmf'
    nameWbasis = folderBasis +  'Wbasis.hd5'
    paramRVEname = folderPrediction + 'paramRVEdataset_validation.hd5'
    bcs_namefile = folderPrediction + 'bcs_{0}_{1}.hd5'.format(archId, Nrb)
    nameScaleXY_shear = folderTrain +  'scaler_S_{0}.txt'.format(Nrb)
    nameScaleXY_axial = folderTrain +  'scaler_A_{0}.txt'.format(Nrb)
    
    net = standardNets[archId]
    
    net.param['nY'] = Nrb
    net.param['nX'] = nX
    net.param['file_weights_shear'] = folderTrain + 'models_weights_{0}_S_{1}.hdf5'.format(archId, Nrb)
    net.param['file_weights_axial'] = folderTrain + 'models_weights_{0}_A_{1}.hdf5'.format(archId, Nrb)

    namefiles = [nameMeshRefBnd, nameWbasis, paramRVEname, nameScaleXY_shear, nameScaleXY_axial]
    
    predictBCs(namefiles, net)


                                                                               