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
# import plotUtils as plut
import myHDF5 as myhd

import matplotlib.pyplot as plt
import copy

listNames = ['LHS_modified','LHS_frozen_p2', 'LHS_frozen_p4']
listNames2 = [r'LHSmodified',r'LHSfrozen_p2', r'LHS_frozen_p4']

folder = ["/Users", "/home"][1] + "/felipefr/switchdrive/scratch/deepBoundary/smartGeneration/{0}/"
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


# myhd.loadhd5(folder + 'eigens.hd5','eigenvalues', mode = 'r')

for l in listNames:
    # lambdas[l] = np.loadtxt('lambda_{0}.txt'.format(l))
    lambdas[l] = myhd.loadhd5(folder.format(l) + 'eigens.hd5','eigenvalues')
    
plt.figure(1)
for l,l2 in zip(listNames,listNames2):
    plt.plot(lambdas[l][:40], label = l2)

plt.legend()
plt.xlabel('N')
plt.ylabel('eigenvalue')
plt.grid()
plt.yscale('log')

plt.savefig('cutSpectrum_new.pdf')