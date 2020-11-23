import sys, os
from numpy import isclose
from dolfin import *
# import matplotlib.pyplot as plt
from multiphenics import *
sys.path.insert(0, '../../utils/')

import fenicsWrapperElasticity as fela
import matplotlib.pyplot as plt
import numpy as np
import generatorMultiscale as gmts
import meshUtils as meut
import generationInclusions as geni
import myCoeffClass as coef
import fenicsMultiscale as fmts
import elasticity_utils as elut
import fenicsWrapperElasticity as fela
import multiphenicsMultiscale as mpms
import ioFenicsWrappers as iofe
import fenicsUtils as feut
import plotUtils as plut
import myHDF5 as myhd

import matplotlib.pyplot as plt
import copy

listNames = ['LHS_maxmin','LHS', 'Sobol', 'random', 'LHS_maxmin_full']
listNames2 = [r'LHSmaxmin',r'LHS', r'Sobol', r'random',r'LHSmaxminfull']

folder = ["/Users", "/home"][0] + "/felipefr/switchdrive/scratch/deepBoundary/smartGeneration/{0}/"
nameC = {}
for l in listNames:
    nameC[l] = folder.format(l) + 'Cnew.h5'

lambdas = {}

# for l in listNames[4:5]:
#     C, fC = myhd.loadhd5_openFile(nameC[l],'C')

#     sig, U = np.linalg.eigh(C)
   
#     asort = np.argsort(sig)
#     sig = sig[asort[::-1]]
#     lambdas[l] = sig
    
#     np.savetxt('lambda_{0}.txt'.format(l),lambdas[l])
#     fC.close()


for l in listNames:
    lambdas[l] = np.loadtxt('lambda_{0}.txt'.format(l))
    
plt.figure(1)
for l,l2 in zip(listNames,listNames2):
    plt.plot(lambdas[l][:160], label = l2)

plt.legend()
plt.xlabel('N')
plt.ylabel('eigenvalue')
plt.grid()
plt.yscale('log')

plt.savefig('allSpectrum.pdf')