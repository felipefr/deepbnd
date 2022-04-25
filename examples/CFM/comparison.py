#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 12:06:50 2022

@author: felipe
"""

import os, sys
import matplotlib.pyplot as plt
import numpy as np
import dolfin as df

from deepBND.__init__ import *
import deepBND.core.data_manipulation.wrapper_h5py as myhd

sym = lambda X: 0.5*(X + X.T)
folder = rootDataPath + '/CFM/dataset/'

cases = ['dnn', 'per', 'lin', 'MR', 'HF']


snapshotsName = folder + 'snapshots_subdomains_{0}.hd5'

snapshotsFullName = folder + 'snapshots_full.hd5'

# COMPARISON HF/reduced models , 6x6 -> 2x2
tangents = {}
id_ = {}
id_local_subdomains = {}
for case in cases:
    tangents[case] = myhd.loadhd5(snapshotsName.format(case), 'tangentL')
    id_[case] = myhd.loadhd5(snapshotsName.format(case), 'id')
    id_local_subdomains[case] = myhd.loadhd5(snapshotsName.format(case), 'id_local')
    
refCase = 'HF'
errors = {}
errors_rel = {}
ns = 90

for case in cases[:-1]:
    errors[case] = np.zeros(ns)
    errors_rel[case] = np.zeros(ns)
    for i in range(ns):
        errors[case][i] = np.linalg.norm(sym(tangents[case][i] - tangents[refCase][i]))
        errors_rel[case][i] = errors[case][i]/np.linalg.norm(sym(tangents[refCase][i]))
        
# for case in cases[:-1]:
#     print(case, np.mean(errors[case]), np.std(errors[case]), np.max(errors[case]))

print("relative Error")
for case in cases[:-1]:
    print(case, np.mean(errors_rel[case]), np.std(errors_rel[case]), np.max(errors_rel[case]), np.mean(errors_rel[case]/errors_rel['dnn']))



# COMPARISON Full HF/Domain-decomposition reduced models , 6x6 -> 9x(2x2) pieces

NS = 10

# Computing the average tangents
tangents_avg = {}
for case in cases:
    tangents_avg[case] = np.zeros((NS,3,3))
    
    for i in range(NS):
        selected_ids = np.where(id_[case] == i)
    
        for j in range(3):
            for k in range(3):
                tangents_avg[case][i,j,k] = np.mean(tangents[case][selected_ids,j,k])
                
                
tangents_full = myhd.loadhd5(snapshotsFullName, 'tangentL')

refCase = 'full'
errors = {}
errors_rel = {}

for case in cases:
    errors[case] = np.zeros(NS)
    errors_rel[case] = np.zeros(NS)
    for i in range(NS):
        errors[case][i] = np.linalg.norm(sym(tangents_avg[case][i] - tangents_full[i]))
        errors_rel[case][i] = errors[case][i]/np.linalg.norm(sym(tangents_full[i]))


print("relative Error")
for case in cases:
    print(case, np.mean(errors_rel[case]), np.std(errors_rel[case]), np.max(errors_rel[case]), np.mean(errors_rel[case]/errors_rel['HF']))

