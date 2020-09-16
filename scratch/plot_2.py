import sys, os
from numpy import isclose
from dolfin import *
# import matplotlib.pyplot as plt
from multiphenics import *
sys.path.insert(0, '../utils/')

import fenicsWrapperElasticity as fela
import matplotlib.pyplot as plt
import numpy as np
import generatorMultiscale as gmts
import wrapperPygmsh as gmsh
import generationInclusions as geni
import myCoeffClass as coef
import fenicsMultiscale as fmts
import elasticity_utils as elut
import fenicsWrapperElasticity as fela
import multiphenicsMultiscale as mpms
import fenicsUtils as feut

from timeit import default_timer as timer

Nbasis = np.array([  1,   4,   9,  16,  26,  38,  52,  69,  88, 109, 133, 159])          

normU = np.array([8.53116375e-03, 7.76108489e-03, 5.15435132e-03, 4.00022477e-03,
       3.44153277e-03, 2.99385755e-03, 2.30837693e-03, 1.07886961e-03,
       7.91741680e-04, 3.93427764e-04, 2.83094849e-04, 4.97216503e-13])

sigmaList = np.array([0.16823522496007692,
 0.16510811456730298,
 0.16556624645282927,
 0.16695079036734767,
 0.16738231110185386,
 0.16765768823590035,
 0.1678181892580636,
 0.16791283352146083,
 0.16812281887036173,
 0.1681588216816549,
 0.16821105565363048,
 0.1682151772912903,
 0.16823522496010565])

                  
plt.figure(1)
plt.plot(Nbasis, normU, '-o')
plt.yscale('log')
plt.ylabel('Norm on boundary: Ref - POD(N)')
plt.xlabel('N')
plt.grid()

plt.figure(2)
plt.plot(Nbasis, sigmaList[1:], '-o')
plt.plot([Nbasis[0],Nbasis[-1]], 2*[sigmaList[0]], label = 'ref')
plt.ylabel('sigma_11')
plt.xlabel('N')
plt.grid()


S = np.loadtxt('snapshots.txt')

Wbasis, sig, Vh = np.linalg.svd(S)
                                       
cumsum = np.cumsum(sig*sig)

plt.figure(3)
plt.plot(np.arange(1,160),1-(cumsum[:-1]/cumsum[-1]), '-o' )
plt.yscale('log')
plt.ylabel('eps POD')
plt.xlabel('N : Number of basis functions')
plt.grid()
plt.show()
