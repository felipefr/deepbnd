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

import matplotlib.pyplot as plt
import copy

l = [ int(sys.argv[i]) for i in range(1,5) ]

folder = ["/Users", "/home"][0] + "/felipefr/EPFL/newDLPDEs/DATA/deepBoundary/convergenceStudy/" 


polynomial = ['P1','P2']
levelRandomness = ['not','partial','total']
refinement = ['', 'Refined']
loading = ['axial','shear']

folderAdd = '{0}{1}Random{2}/{3}/'.format(polynomial[l[0]],levelRandomness[l[1]],refinement[l[2]],loading[l[3]])

folder += folderAdd

# loading = 'Shear-P1-partialRandom'
label = '{0}-{1} Random-{2}({3})'.format(polynomial[l[0]],levelRandomness[l[1]],refinement[l[2]],loading[l[3]])
filelabel = 'Error_{0}_{1}_Random_{2}_{3}.pdf'.format(polynomial[l[0]],levelRandomness[l[1]],refinement[l[2]],loading[l[3]])


# if(loading == 'Shear'):
#     folder += 'shear/'

Offset0 = 0
maxOffset = 7

NxL = 2
Lratio = 1.0 + 2*np.arange(maxOffset + 1)/NxL

Nseed = 20
seed0 = 0

BCs = ['periodic','MR','Lin']

sigma = {}
sigma['L'] = {}
sigma['T'] = {}
for model in BCs:
    sigma['L'][model] = np.zeros((Nseed,maxOffset + 1,4))
    sigma['T'][model] = np.zeros((Nseed,maxOffset + 1,4))

for model in BCs:
    for i in range(Nseed):
        for j in range(maxOffset+1): 
            sigma['L'][model][i,j] = np.loadtxt(folder + 'sigmaL_{0}_offset{1}_{2}.txt'.format(model,j,i + seed0))
            sigma['T'][model][i,j] = np.loadtxt(folder + 'sigmaT_{0}_offset{1}_{2}.txt'.format(model,j,i + seed0))

# sigmaRefL = 0.5*( sigma['L']['Lin'][:,-1,:] + sigma['L']['periodic'][:,-1,:])  
# sigmaRefT = 0.5*( sigma['T']['Lin'][:,-1,:] + sigma['L']['periodic'][:,-1,:])   


# getError = lambda X, x0: np.array( [ [ np.linalg.norm(X[j,i,:] - x0[j,:])/np.linalg.norm(x0[j,:])    for i in range(len(X[0]))]  for j in range(len(X)) ]) # j runs in seeds, i runs in offset
getErrorInc = lambda X : np.array( [ [ np.linalg.norm(X[j,i,:] - X[j,i+1,:])/np.linalg.norm(X[j,i+1,:])    for i in range(len(X[0])-1)]  for j in range(len(X)) ]) # j runs in seeds, i runs in offset
getError = lambda X : np.array( [ [ np.linalg.norm(X[j,i,:] - X[j,-1,:])/np.linalg.norm(X[j,-1,:])    for i in range(len(X[0])-1)]  for j in range(len(X)) ]) # j runs in seeds, i runs in offset



error = {}
error['L'] = {}
error['T'] = {}

errorInc = {}
errorInc['L'] = {}
errorInc['T'] = {}

for model in BCs:
    # error['L'][model] = getError( sigma['L'][model], sigmaRefL)
    # error['T'][model] = getError( sigma['T'][model], sigmaRefT)
    errorInc['L'][model] = getErrorInc( sigma['L'][model])
    errorInc['T'][model] = getErrorInc( sigma['T'][model])
    error['L'][model] = getError( sigma['L'][model])
    error['T'][model] = getError( sigma['T'][model])
    

modelLabel = {'periodic': r"\Large $V_{\mu}^{P,\Omega'_{\mu}}$" , 'MR': r"\Large $V_{\mu}^{M,\Omega'_{\mu}}$", 'Lin': r"\Large $V_{\mu}^{L,\Omega'_{\mu}}$" }


plut.palletteCounter = 0
plt.figure(3, (6,4))
for model in BCs:
    plut.plotFillBetweenStd( Lratio[:-1], errorInc['L'][model][:,:] , l = modelLabel[model])

plt.xlabel(r"\Large $L'_{\mu}/L_{\mu}$")
plt.ylabel(r"\Large $E_{IR}(L_{\mu}')$")
plt.title(r" \Large Stress Incremental Relative Error ({0})".format(label))

plt.yscale('log')

plt.legend(loc = 'best')
plt.grid()

# plt.plot(Lratio[:-1], 1.0e-4*np.ones(7), "m--")

plt.savefig("incremental" + filelabel)


plut.palletteCounter = 0
plt.figure(4, (6,4))
for model in BCs:
    plut.plotFillBetweenStd( Lratio[:-1], error['L'][model][:,:] , l = modelLabel[model])

plt.xlabel(r"\Large $L'_{\mu}/L_{\mu}$")
plt.ylabel(r"\Large $E_{rel}(L_{\mu}')$")
plt.title(r" \Large Stress Relative Error ({0})".format(label))

plt.yscale('log')

plt.legend(loc = 'best')
plt.grid()

# plt.plot(Lratio[:-1], 1.0e-4*np.ones(7), "m--")

plt.savefig(filelabel)

# plt.show()