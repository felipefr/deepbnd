#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 16:04:12 2022

@author: felipe
"""

import os, sys
import matplotlib.pyplot as plt
import numpy as np
import dolfin as df

from deepBND.__init__ import *
import deepBND.core.data_manipulation.wrapper_h5py as myhd

# Total time for 100x(10x10) 2845.171440158003s

timeFull = 28.4517
timeReduced = 0.5
timeHF = 4.8

sym = lambda X: 0.5*(X + X.T)
folder = rootDataPath + '/CFM/dataset_NS100/'

cases = ['dnn', 'HF_large', 'HF']


snapshotsName = folder + 'snapshots_subdomains_{0}.hd5'

snapshotsFullName = folder + 'snapshots_full.hd5'

# COMPARISON HF/reduced models , 6x6 -> 2x2
tangents = {}
eigenvalues = {}
id_ = {}
id_local_subdomains = {}
for case in cases:
    if(case == 'HF_large'):
        tangents[case] = myhd.loadhd5(snapshotsName.format('HF'), 'tangent')
        eigenvalues[case] = np.array( [ np.linalg.eig(sym(tangents[case][i]))[0] for i in range(len(tangents[case])) ] ) 
        id_[case] = myhd.loadhd5(snapshotsName.format('HF'), 'id')
        id_local_subdomains[case] = myhd.loadhd5(snapshotsName.format('HF'), 'id_local')
        
    else:
        tangents[case] = myhd.loadhd5(snapshotsName.format(case), 'tangentL')
    
        eigenvalues[case] = np.array( [ np.linalg.eig(sym(tangents[case][i]))[0] for i in range(len(tangents[case])) ] ) 
        id_[case] = myhd.loadhd5(snapshotsName.format(case), 'id')
        id_local_subdomains[case] = myhd.loadhd5(snapshotsName.format(case), 'id_local')
        
refCase = 'HF'
errors = {}
errors_rel = {}
ns = 900

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

NS = 100

# Computing the average tangents
tangents_avg = {}
for case in cases:
    tangents_avg[case] = np.zeros((NS,3,3))
    
    for i in range(NS):
        selected_ids = np.where(id_[case] == i)
    
        for j in range(3):
            for k in range(3):
                tangents_avg[case][i,j,k] = np.mean(tangents[case][selected_ids,j,k])
                
                
tangents_full = myhd.loadhd5(snapshotsFullName, 'tangent')
eigenvalues_full = np.array( [ np.linalg.eig(sym(tangents_full[i]))[0] for i in range(len(tangents_full)) ])  

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


# Analysing confidence interval

# Component
ind1 = 1
ind2 = 1

avg_N = np.zeros(NS)
std_N = np.zeros(NS)
cfi_N = np.zeros(NS)

timeFull_N = np.linspace(timeFull, NS*timeFull, NS)


np.random.seed(3)
shuffled_ids = np.arange(0, NS)
np.random.shuffle(shuffled_ids)

for i in range(NS):
    avg_N[i] = np.mean( tangents_full[ shuffled_ids[: i+1], ind1, ind2] )
    std_N[i] = np.std( tangents_full[ shuffled_ids[: i+1], ind1, ind2] )
    cfi_N[i] = 1.96*std_N[i]/(np.sqrt(i+1)*avg_N[i]) # i + 1 is the actual number    

avg_subdomains_N = {}
std_subdomains_N = {}
cfi_subdomains_N = {}

np.random.seed(8)
shuffled_ids = np.arange(0, ns)
np.random.shuffle(shuffled_ids)

timeReduced_N = np.linspace(timeReduced, ns*timeReduced, ns)

for case in cases:
    avg_subdomains_N[case] = np.zeros(ns)
    std_subdomains_N[case] = np.zeros(ns)
    cfi_subdomains_N[case] = np.zeros(ns)

    for i in range(ns):
        avg_subdomains_N[case][i] = np.mean( tangents[case][ shuffled_ids[: i+1], ind1, ind2] )
        std_subdomains_N[case][i] = np.std( tangents[case][ shuffled_ids[: i+1], ind1, ind2] )
        cfi_subdomains_N[case][i] = 1.96*std_subdomains_N[case][i]/(np.sqrt(i+1)*avg_subdomains_N[case][i]) # i + 1 is the actual number
        

    
NS0 = 40
ns0 = 360
fac = NS/ns
    
plt.figure(1)
plt.plot(np.arange(NS0, NS) + 1, avg_N[NS0:], '-o', label = 'full')
for case in cases:
    plt.plot(fac*(np.arange(ns0, ns) + 1), avg_subdomains_N[case][ns0:], '-o', label = case)

plt.legend()
plt.grid()

plt.figure(2)
plt.plot(np.arange(NS0, NS) + 1, cfi_N[NS0:], '-o', label = 'full')
for case in cases:
    plt.plot(fac*(np.arange(ns0, ns) + 1), cfi_subdomains_N[case][ns0:], '-o', label = case)

plt.legend()
plt.grid()



plt.figure(3)
plt.plot(np.arange(NS0, NS) + 1, avg_N[NS0:], '-ro', label = 'full')
plt.plot(np.arange(NS0, NS) + 1, avg_N[NS0:]*(1.0 + cfi_N[NS0:]), '--r', label = 'full + cfi')
plt.plot(np.arange(NS0, NS) + 1, avg_N[NS0:]*(1.0 - cfi_N[NS0:]), '--r', label = 'full - cfi')

for case, c in zip(cases, ['b', 'g', 'y']):
    plt.plot(fac*(np.arange(ns0, ns) + 1), avg_subdomains_N[case][ns0:], c + '-o', label = case)
    plt.plot(fac*(np.arange(ns0, ns) + 1), avg_subdomains_N[case][ns0:]*(1.0 + cfi_subdomains_N[case][ns0:]), c + '--', label = case + '+cfi')
    plt.plot(fac*(np.arange(ns0, ns) + 1), avg_subdomains_N[case][ns0:]*(1.0 - cfi_subdomains_N[case][ns0:]), c + '--', label = case + '-cfi')
    
plt.legend()
plt.grid()

# plt.figure(3)
# plt.plot(timeFull_N[NS0:], avg_N[NS0:], '-o')
# for case in cases:
#     plt.plot(timeReduced_N[ns0:], avg_subdomains_N[case][ns0:], '-o', label = case)

# plt.legend()
# plt.grid()

# plt.figure(4)
# plt.plot(timeFull_N[NS0:], cfi_N[NS0:], '-o')
# for case in cases:
#     plt.plot(timeReduced_N[ns0:], cfi_subdomains_N[case][ns0:], '-o', label = case)

# plt.legend()
# plt.grid()
