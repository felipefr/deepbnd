#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 14:01:51 2022

@author: felipefr
"""

import os, sys
import matplotlib.pyplot as plt
import numpy as np
import dolfin as df

from deepBND.__init__ import *
import fetricks.data_manipulation.wrapper_h5py as myhd

# Test Loading 
problemType = ''

folder = rootDataPath + '/review2_smaller/'
folderTangent = folder + '/prediction_test/'

cases = ['old', 'new']

# loading cases
ns = 10
tangentName = folderTangent + 'tangents_{0}.hd5'
tangents = {}
for case in cases:
    tangents[case] = myhd.loadhd5(tangentName.format(case), 'tangent')[:ns]


refCase = 'old'
errors = {}

for case in cases[1:]:
    errors[case] = np.zeros(ns)
    for i in range(ns):
        errors[case][i] = np.linalg.norm(tangents[case][i] - tangents[refCase][i])
        
for case in cases[1:]:
    print(case, np.mean(errors[case]), np.std(errors[case]))
