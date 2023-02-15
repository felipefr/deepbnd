#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 10:44:19 2022

@author: felipe
"""

"""
This file is part of deepBND, a data-driven enhanced boundary condition implementaion for 
computational homogenization problems, using RB-ROM and Neural Networks.
Copyright (c) 2020-2023, Felipe Rocha.
See file LICENSE.txt for license information. 
Please cite this work according to README.md.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or <felipe.f.rocha@gmail.com>
"""

import sys, os
import dolfin as df
import numpy as np

from deepBND.__init__ import *
from fetricks.fenics.mesh.mesh import Mesh 
from fetricks.mechanics.piola import PiolaTransform_rotation_matricial
from deepBND.creation_model.prediction.NN_elast import NNElast
import deepBND.creation_model.dataset.generation_inclusions as geni
from deepBND.core.multiscale.mesh_RVE import paramRVE_default 

class NNElast_positions(NNElast):
    def predict(self, X, Vref): # X = [cos(theta), sin(theta)] 
        
        p = paramRVE_default(NxL = 4, NyL = 4, maxOffset = 2, Vfrac = 0.0) # Vfrac is dummy
    
        paramEllipses = geni.getEllipse_emptyRadius(p.Nx,p.Ny,p.Lxt, p.Lyt, p.x0, p.y0)
    
        X0 = paramEllipses[:,0:2].flatten()
        
        delta = np.zeros(X.shape)
        for i in range(len(delta)):
            delta[i,:] = X[i,:] - X0 
        
        X_s = {}  
        Y_p = {}; S_p = {}
        
        for l in self.labels:
            X_s[l] = self.scalerX[l].transform(X)
            Y_p[l] = self.scalerY[l].inverse_transform(self.model[l].predict(X_s[l]))
            S_p[l] = Y_p[l] @ self.Wbasis[l][:self.Nrb,:]

        
        # Y here doesn't mean the output but the vertical direction
        X_axialY = np.zeros(X.shape)
        delta_x = delta[:, 0::2]
        delta_y = delta[:, 1::2]
        
        perm = self.getPermY(8)
        
        for i in range(len(delta)):
            X_axialY[i, 0::2] = X0[0::2] + delta_y[i,perm] # x component
            X_axialY[i, 1::2] = X0[1::2] - delta_x[i,perm] # y component
        
        X_axialY_s = self.scalerX['A'].transform(X_axialY) 
        Y_p_axialY = self.scalerY['A'].inverse_transform(self.model['A'].predict(X_axialY_s)) 
        
        theta = 3*np.pi/2.0 # 'minus HalfPi'
        piola_mat = PiolaTransform_rotation_matricial(theta, Vref)
        
        print(piola_mat.shape)
        print(self.Wbasis['A'].shape, self.Nrb)
        print(Y_p_axialY.shape)
        S_p_axialY = Y_p_axialY @ self.Wbasis['A'][:self.Nrb,:] @ piola_mat.T #changed
        
        return [S_p['A'], S_p_axialY, S_p['S']]
    
    def getPermY(self, n = 6):  
        # the permY below is only valid for the radius ordenated by rows and columns (below to top {left to right})
        return np.array([[(n-1-j)*n + i for j in range(n)] for i in range(n)]).flatten() # note that (i,j) -> (Nx-j-1,i)
