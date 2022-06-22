#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 08:56:10 2022

@author: felipe
"""

import os, sys
import matplotlib.pyplot as plt
import numpy as np
import dolfin as df

from deepBND.__init__ import *
import deepBND.core.data_manipulation.wrapper_h5py as myhd

# Test Loading 
problemType = ''

folder = rootDataPath + '/review2/cook_fresh_test/meshes_seed%d/'
folderTangent = rootDataPath + '/review2/prediction/'

cases = ['dnn', 'per', 'per_full_old']

tangentName = folderTangent + 'tangents_{0}.hd5'
tangents = {}
for case in cases:
    tangents[case] = myhd.loadhd5(tangentName.format(case), 'tangent')


refCase = 'per_full_old'
errors = {}
ns = 20
for case in cases[:-1]:
    errors[case] = np.zeros(ns)
    for i in range(ns):
        errors[case][i] = np.linalg.norm(tangents[case][i] - tangents[refCase][i])
        
for case in cases[:-1]:
    print(case, np.mean(errors[case]), np.std(errors[case]))
