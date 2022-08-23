import sys, os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import dolfin as df
import numpy as np

from deepBND.__init__ import *
import deepBND.core.data_manipulation.utils as dman
import deepBND.core.data_manipulation.wrapper_h5py as myhd
from deepBND.core.fenics_tools.enriched_mesh import EnrichedMesh 
import deepBND.core.elasticity.fenics_utils as feut

class NNElast:
    
    def __init__(self, nameWbasis, net, Nrb, scalertype = 'MinMax'):
        
        self.Nrb = Nrb
        self.labels = net.keys()
        
        self.Wbasis = {}
        self.scalerX = {}; self.scalerY = {}
        self.model = {}

        for l in self.labels:
            self.Wbasis[l] = myhd.loadhd5(nameWbasis, 'Wbasis_%s'%l)
            self.scalerX[l], self.scalerY[l]  = dman.importScale(net[l].files['scaler'], net[l].nX, net[l].nY, scalertype)
            self.model[l] = net[l].getModel()
            self.model[l].load_weights(net[l].files['weights'])

    def predict(self, X, Vref):
        X_s = {}  
        Y_p = {}; S_p = {}
        
        for l in self.labels:
            X_s[l] = self.scalerX[l].transform(X)
            Y_p[l] = self.scalerY[l].inverse_transform(self.model[l].predict(X_s[l]))
            S_p[l] = Y_p[l] @ self.Wbasis[l][:self.Nrb,:]

        X_axialY_s = self.scalerX['A'].transform(X[:,self.getPermY()]) ### permY performs a counterclockwise rotation
        Y_p_axialY = self.scalerY['A'].inverse_transform(self.model['A'].predict(X_axialY_s)) 
        
        
        theta = 3*np.pi/2.0 # 'minus HalfPi'
        piola_mat = feut.PiolaTransform_rotation_matricial(theta, Vref)
        S_p_axialY = Y_p_axialY @ self.Wbasis['A'][:self.Nrb,:] @ piola_mat.T #changed
        
        return [S_p['A'], S_p_axialY, S_p['S']]
        
    def getPermY(self):  
        ## it may be changed (permY)
        # the permY below is only valid for the ordenated radius (inside to outsid)
        # permY = np.array([2,0,3,1,12,10,8,4,13,5,14,6,15,11,9,7,30,28,26,24,22,16,31,17,32,18,33,19,34,20,35,29,27,25,23,21])
        # the permY below is only valid for the radius ordenated by rows and columns (below to top {left to right})
        return np.array([[(5-j)*6 + i for j in range(6)] for i in range(6)]).flatten() # note that (i,j) -> (Nx-j-1,i)
        