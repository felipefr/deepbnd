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
import myHDF5 as myhd
import matplotlib.pyplot as plt
import copy


folder = ["/Users", "/home"][0] + "/felipefr/switchdrive/scratch/deepBoundary/RBsensibility/fullSampled/"

# tension test
epsTension = np.zeros((2,2))
epsTension[0,0] = 1.0 # after rescaled to have 1.0

# shear test
epsShear = np.zeros((2,2))
epsShear[0,1] = 0.5
epsShear[1,0] = 0.5

maxOffset = 4

H = 1.0 # size of each square
NxL = NyL = 2
NL = NxL*NyL
x0L = y0L = -H 
LxL = LyL = 2*H
lcar = (2/30)*H
Nx = (NxL+2*maxOffset)
Ny = (NyL+2*maxOffset)
Lxt = Nx*H
Lyt = Ny*H
NpLxt = int(Lxt/lcar) + 1
NpLxL = int(LxL/lcar) + 1
print("NpLxL=", NpLxL) 
x0 = -Lxt/2.0
y0 = -Lyt/2.0
r0 = 0.2*H
r1 = 0.4*H
Vfrac = 0.282743
rm = H*np.sqrt(Vfrac/np.pi)

ns = 1000
seed = 1
snapshots= {}
fsnaps = {}
snap_solutions = {}
snap_sigmas = {}
snap_a = {}
snap_B = {}

# for opLoad in ['axial','shear']:
#     for opModel in ['MR','Lin']:
#         os.system('rm ' + folder +  '{0}/{1}/snapshots_{1}.h5'.format(opModel,opLoad,seed))
#         snapshots[opModel + opLoad], fsnaps[opModel + opLoad] = myhd.zeros_openFile(filename = folder +  '{0}/{1}/snapshots_{1}.h5'.format(opModel,opLoad,seed),  
#                                                 shape = [(ns,Vref.dim()),(ns,3),(ns,2),(ns,2,2)], label = ['solutions','sigma','a','B'], mode = 'w-')
        
#         snap_solutions[opModel + opLoad], snap_sigmas[opModel + opLoad], snap_a[opModel + opLoad], snap_B[opModel + opLoad] = snapshots[opModel + opLoad]

   
# ellipseData = myhd.loadhd5(folder +  'ellipseData_{0}.h5'.format(seed), 'ellipseData')


# for i in range(10):
#     x = ellipseData[i,:,0]
#     y = ellipseData[i,:,1]
#     r = ellipseData[i,:,2]
    
#     fig = plt.figure(i,(8,8))
    
#     ax = fig.add_subplot(1,1,1)
    
#     ax.set(xlim=(-0.5*Lxt, 0.5*Lxt), ylim = (-0.5*Lyt, 0.5*Lyt))
    
#     for j in range(len(x)):
#         a_circle = plt.Circle((x[j], y[j]), r[j])
#         ax.add_artist(a_circle)
        
#     fig.subplots_adjust(bottom = 0)
#     fig.subplots_adjust(top = 1)
#     fig.subplots_adjust(right = 1)
#     fig.subplots_adjust(left = 0)
    
#     plt.axis('off')
    
#     fig.savefig('out_{0}.png'.format(i), bbox_inches='tight', pad_inches=0)

img = []
imgfft = []
F2 = []
for i in range(10):
    img.append(plt.imread('out_{0}.png'.format(i)))
    img[-1] = np.dot(img[-1][:,:,:3],np.ones(3)/3.0)
    # img[-1] = 1.0 - img[-1]
    imgfft.append(np.fft.fft2(img[-1]))
    F2.append(np.abs(imgfft[-1])**2.0/(2.0*np.pi*len(imgfft[-1][0])*len(imgfft[-1])))
    # plt.figure(i)
    # plt.imshow(img[-1],cmap = 'gray')       
    plt.figure(i)
    plt.imshow(np.log(F2[-1][:,:]))
    plt.colorbar()
    plt.savefig('fft_{0}.png'.format(i))
          
