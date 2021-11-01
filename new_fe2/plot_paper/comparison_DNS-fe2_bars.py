#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 11:52:26 2021

@author: felipefr
"""

import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0,'../../utils/')

# import generation_deepBoundary_lib as gdb
import matplotlib.pyplot as plt
import numpy as np

# import myTensorflow as mytf
from timeit import default_timer as timer

import h5py
import myHDF5 as myhd
import matplotlib.pyplot as plt
import meshUtils as meut
from dolfin import *


import plotUtils

# Test Loading 

rootDataPath = open('../../../rootDataPath.txt','r').readline()[:-1]
print(rootDataPath)

Ny_DNS = 24
typeProblem = 'leftClamped'

folder = rootDataPath + '/new_fe2/DNS/DNS_%d_new/'%Ny_DNS

tangent_labels = ['DNS', 'DeepBND', 'Periodic', 'High-fidelity']
solutions = [ 'barMacro_DNS.xdmf', 
             'multiscale/barMacro_Multiscale_dnn_big.xdmf',
             'multiscale/barMacro_Multiscale_reduced_per.xdmf',
              'multiscale/barMacro_Multiscale_full.xdmf'] # temporary
         
solutions = [folder + f for f in solutions]

meshDNSfile =  folder + 'mesh.xdmf'
meshMultiscaleFile = folder + 'multiscale/meshBarMacro_Multiscale.xdmf'

meshMultiscale = Mesh()
with XDMFFile(meshMultiscaleFile) as infile:
    infile.read(meshMultiscale)
        

Uh_mult = VectorFunctionSpace(meshMultiscale, "CG", 2)

uhs = []
for f in solutions:    
    uh = Function(Uh_mult)
    print(f)
    with XDMFFile(f) as infile:
        infile.read_checkpoint(uh, 'u', 0)
        uhs.append(uh)

# uhs[0] = interpolate(uhs[0],Uh_mult)

if(typeProblem == 'rightClamped'):
    pA = Point(0.0,0.0)
    pB = Point(0.0,0.5)
elif(typeProblem == 'bending'):
    pA = Point(0.5,0.0)
    pB = Point(0.5,0.5)
else: 
    pA = Point(2.0,0.0)
    pB = Point(2.0,0.5)

# norms_label = ['$\|\cdot\|_{$L^2(\Omega)$}', '$\|\cdot(\Vec{x}=\Vec{x}_A)\|_2$', '$\|\cdot(\Vec{x}=\Vec{x}_B)\|_2$']
norms_label = [r'$c$}', r'$b$', r'$a$']
norms = [lambda x: norm(x), lambda x: np.linalg.norm(x(pA)), lambda x: np.linalg.norm(x(pB))]
norms_ref = np.array([N(uhs[0]) for N in norms]) 
 

e = Function(Uh_mult)
n = len(solutions) - 1
errors = np.zeros((n,len(norms)))
errors_rel = np.zeros((n,len(norms)))


for i in range(n):
    e.vector().set_local(uhs[i+1].vector().get_local()[:]-uhs[0].vector().get_local()[:])
    for j, N in enumerate(norms):
        errors[i,j] = N(e)
    
    errors_rel[i,:] = errors[i,:]/norms_ref[:]
    

# np.savetxt(folder + 'errors.txt', errors)

suptitle = 'Error_DNS_%d_bar_vs_DNS'%Ny_DNS 

# np.savetxt(folder + 'errors_{0}.txt'.format(suptitle), errors)


labels = ['$\|(\cdot)\|_{L^2(\Omega)}$', '$\|(\cdot)(\mathbf{x}_A)\|_2$', '$\|(\cdot)(\mathbf{x}_B)\|_2$']
x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

plt.figure(1,(5.5,3.5))

fig, ax = plt.subplots()
for i in range(n):
    ax.bar(x - (i-1)*width, errors[i,:], width, label= tangent_labels[i+1])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_yscale('log')
ax.set_ylim(0.4e-2,1.1e-2)
ax.set_xlabel('Norms')
ax.set_ylabel('Error')
ax.set_title('Absolute Error against DNS solution ($N_y^{DNS} = %d$)'%Ny_DNS)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc=2)
fig.tight_layout()
ax.grid()

plt.savefig('errordns_ny{0}_bars.pdf'.format(Ny_DNS))
plt.savefig('errordns_ny{0}_bars.eps'.format(Ny_DNS))

plt.figure(2,(5.5,3.5))

fig, ax = plt.subplots()
for i in range(n):
    ax.bar(x - (i-1)*width, errors_rel[i,:], width, label= tangent_labels[i+1])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_yscale('log')
ax.set_ylim(0.9e-2,1.2e-2)
ax.set_xlabel('Norms')
ax.set_ylabel('Error')
ax.set_title('Relative Error against DNS solution ($N_y^{DNS} = %d$)'%Ny_DNS)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc=2)
fig.tight_layout()
ax.grid()

plt.savefig('error_rel_ny{0}_bars.pdf'.format(Ny_DNS))
plt.savefig('error_rel_ny{0}_bars.eps'.format(Ny_DNS))



