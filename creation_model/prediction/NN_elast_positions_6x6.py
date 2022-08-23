#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 15:08:22 2022

@author: felipefr
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
from fetricks.fenics.misc import affineTransformationExpression 

import fetricks as ft


# Converse convention than adopted in the original paper:
# In the original paper: halfPi permutation clockwise and halfpi rotation anticlockwise  
# Implementation: halfPi permutation anticlockwise and halfpi rotation clockwise (-Pi anticlockwise)

class NNElast_positions_6x6(NNElast):
    def predict(self, X, Vref): # X = [ (x_i, y_i)]_i=1^36
        
        NL = 2
        maxOffset = 2
        Ntot = NL + 2*maxOffset
        p = paramRVE_default(NxL = NL, NyL = NL, maxOffset = maxOffset, Vfrac = 0.0) # Vfrac is dummy
    
        paramEllipses = geni.getEllipse_emptyRadius(p.Nx,p.Ny,p.Lxt, p.Lyt, p.x0, p.y0)
    
        X0 = paramEllipses[:,0:2].flatten()
        
        delta = np.zeros(X.shape)
        print(X.shape, X0.shape)
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
        
        perm = self.getPermY(Ntot)
        
        for i in range(len(delta)):
            X_axialY[i, 0::2] = X0[0::2] - delta_y[i,perm] # x component
            X_axialY[i, 1::2] = X0[1::2] + delta_x[i,perm] # y component
        
        X_axialY_s = self.scalerX['A'].transform(X_axialY) 
        Y_p_axialY = self.scalerY['A'].inverse_transform(self.model['A'].predict(X_axialY_s)) 
        
        theta = 3*np.pi/2.0 # 'minus HalfPi'
        piola_mat = PiolaTransform_rotation_matricial(theta, Vref)
        
        S_p_axialY = Y_p_axialY @ self.Wbasis['A'][:self.Nrb,:] @ piola_mat.T #changed
        
        return [S_p['A'], S_p_axialY, S_p['S']]
    
    
        
    def predict_correctedbyBten(self, X, Vref, Bten_ref):
        
        S_p = self.predict(X, Vref)
        
        uD = df.Function(Vref)   
        ns = len(S_p[0])
        normal = df.FacetNormal(Vref.mesh())
        volL = 4.0 
        
        Bten_ref_Ay = np.array( [ np.array([[Bten_ref['A'][i, 1,1], Bten_ref['A'][i,1,0]], 
                                            [Bten_ref['A'][i,0,1], Bten_ref['A'][i,0,0]]]) for i in range(ns)])

        Bten_ref_list = [ Bten_ref['A'], Bten_ref_Ay, Bten_ref['S']]
        
        for S_p_i, Bten_ref_i in zip(S_p , Bten_ref_list): 
            
            for j in range(ns):
                uD.vector().set_local(S_p_i[j])
                B = -ft.Integral( df.outer(uD, normal), Vref.mesh().ds, (2,2) )/volL
                T = affineTransformationExpression(np.zeros(2), B - Bten_ref_i[j] , Vref.mesh())
                
                S_p_i[j] += df.interpolate(T,Vref).vector().get_local()[:] 
                
        return S_p
        
    def getPermY(self, n): # n^2 is the number of inclusions (in a squared grid)   
        # the permY below is only valid for the radius ordenated by rows and columns (below to top {left to right})
        return np.array([[(n-1-j)*n + i for j in range(n)] for i in range(n)]).flatten() # note that (i,j) -> (Nx-j-1,i)
