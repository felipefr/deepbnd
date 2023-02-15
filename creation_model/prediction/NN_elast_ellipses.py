#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 09:50:56 2022

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
from deepBND.core.fenics_tools.enriched_mesh import EnrichedMesh 
import deepBND.core.elasticity.fenics_utils as feut
from deepBND.creation_model.prediction.NN_elast import NNElast
from deepBND.examples.ellipses.build_rb import mapSinusCosinus

class NNElast_ellipses(NNElast):
    def predict(self, Theta, Vref): # X = [cos(theta), sin(theta)] 
        
        X = mapSinusCosinus(Theta)
        
        X_s = {}  
        Y_p = {}; S_p = {}
        
        for l in self.labels:
            X_s[l] = self.scalerX[l].transform(X)
            Y_p[l] = self.scalerY[l].inverse_transform(self.model[l].predict(X_s[l]))
            S_p[l] = Y_p[l] @ self.Wbasis[l][:self.Nrb,:]

        
        # Y here doesn't mean the output but the vertical direction
        Theta_axialY = Theta[:,self.getPermY()] - 0.5*np.pi ### permY performs a counterclockwise permutation
        X_axialY = mapSinusCosinus(Theta_axialY)
        
        X_axialY_s = self.scalerX['A'].transform(X_axialY) 
        Y_p_axialY = self.scalerY['A'].inverse_transform(self.model['A'].predict(X_axialY_s)) 
        
        theta_rot = 3*np.pi/2.0 # 'minus HalfPi'
        piola_mat = feut.PiolaTransform_rotation_matricial(theta_rot, Vref)
        S_p_axialY = Y_p_axialY @ self.Wbasis['A'][:self.Nrb,:] @ piola_mat.T #changed
        
        return [S_p['A'], S_p_axialY, S_p['S']]
