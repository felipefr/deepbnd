import sys, os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import dolfin as df
import matplotlib.pyplot as plt
from ufl import nabla_div
import numpy as np

from deepBND.__init__ import *
from deepBND.creation_model.training.net_arch import standardNets
import deepBND.core.data_manipulation.utils as dman
import deepBND.core.data_manipulation.wrapper_h5py as myhd
from deepBND.core.fenics_tools.enriched_mesh import EnrichedMesh 
from deepBND.creation_model.prediction.NN_elast import NNElast
    
def predictBCs(namefiles, net):
    
    labels = net.keys()
    
    nameMeshRefBnd, nameWbasis, paramRVEname, bcs_namefile = namefiles
  
    # loading boundary reference mesh
    Mref = EnrichedMesh(nameMeshRefBnd)
    Vref = df.VectorFunctionSpace(Mref,"CG", 1)
    # normal = FacetNormal(Mref)
    # volMref = 4.0
    
    # loading the DNN model
    paramRVEdata = myhd.loadhd5(paramRVEname, 'param')
    
    model = NNElast(nameWbasis, net, net['A'].nY)
    S_p = model.predict(paramRVEdata[:,:,2], Vref)
    
    myhd.savehd5(bcs_namefile, S_p, ['u0','u1','u2'], mode = 'w')

if __name__ == '__main__':
  
    labels = ['A', 'S']
  
    archId = 'small'
    Nrb = 80
    nX = 36
    
    folder = rootDataPath + "/ellipses/"
    folderTrain = folder + 'training_cluster/'
    folderBasis = folder + 'dataset_cluster/'
    folderPrediction = folder + "prediction_cluster/"
    nameMeshRefBnd = folderBasis + 'boundaryMesh.xdmf'
    nameWbasis = folderBasis +  'Wbasis.hd5'
    paramRVEname = folderPrediction + 'paramRVEdataset_validation.hd5'
    bcs_namefile = folderPrediction + 'bcs_{0}_{1}.hd5'.format(archId, Nrb)
    nameScaleXY = {}
    
    net = {}
    
    for l in labels:
        net[l] = standardNets[archId] 
        net[l].nY = Nrb
        net[l].nX = nX
        net[l].files['weights'] = folderTrain + 'models_weights_%s_%s_%d.hdf5'%(archId, l, Nrb)
        net[l].files['scaler'] = folderTrain + 'scaler_%s_%d.txt'%(l, Nrb)

    namefiles = [nameMeshRefBnd, nameWbasis, paramRVEname, bcs_namefile]
    
    predictBCs(namefiles, net)


                                                                               