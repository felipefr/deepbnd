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
# Nbasis = np.arange(150, 161)          


# normU = np.array([8.53116375e-03, 7.76108489e-03, 5.15435132e-03, 4.00022477e-03,
#        3.44153277e-03, 2.99385755e-03, 2.30837693e-03, 1.07886961e-03,
#        7.91741680e-04, 3.93427764e-04, 2.83094849e-04, 4.97216503e-13])

# sigmaList = np.array([0.16823522496007692,
#  0.16510811456730298,
#  0.16556624645282927,
#  0.16695079036734767,
#  0.16738231110185386,
#  0.16765768823590035,
#  0.1678181892580636,
#  0.16791283352146083,
#  0.16812281887036173,
#  0.1681588216816549,
#  0.16821105565363048,
#  0.1682151772912903,
#  0.16823522496010565])

                  
# plt.figure(1)
# plt.plot(Nbasis, normU, '-o')
# plt.yscale('log')
# plt.ylabel('Norm on boundary: Ref - POD(N)')
# plt.xlabel('N')
# plt.grid()

# plt.figure(2)
# plt.plot(Nbasis, sigmaList[1:], '-o')
# plt.plot([Nbasis[0],Nbasis[-1]], 2*[sigmaList[0]], label = 'ref')
# plt.ylabel('sigma_11')
# plt.xlabel('N')
# plt.grid()

condNumber=[8501.523981928842,
            8022.166952788441,
            4386.826189619527,
            4238.730069956054,
            4343.613271434253,
            5775.986235178599,
            9629.178912335055,
            11279.018211504239,
            13629.177898307878,
            18425.77398201267,
            24389.224381212996,
            3.7461535455401667e+18]

condNumber2 = np.loadtxt('conditionNumber.txt')


# condNumber = [25789.754824153006,
#             25885.769387630917,
#             25894.572203016363,
#             25894.573927933867,
#             26129.134678042632,
#             26175.967993762366,
#             26376.70530644786,
#             1.0102469681860475e+19,
#             7.549465061196136e+18,
#             3.7461535455401667e+18,
#             5.907367140151071e+18]


plt.figure(1,(7,5))
plt.plot(Nbasis, condNumber, '-o' , label = 'MR')
plt.plot(Nbasis, condNumber2[1:], '-o' , label = 'periodic')
plt.yscale('log')
plt.ylabel('condNumber')
plt.xlabel('N')
plt.legend()
plt.grid()

plt.show()

S = np.loadtxt('snapshots.txt')

Wbasis, sig1, Vh = np.linalg.svd(S)
                                       
cumsum1 = np.cumsum(sig1*sig1)

S = np.loadtxt('snapshots_periodic.txt')

Wbasis, sig2, Vh = np.linalg.svd(S)

cumsum2 = np.cumsum(sig2*sig2)


plt.figure(3,(8,12  ))
plt.subplot('311')
plt.plot(np.arange(1,161),sig1, '-o' , label = 'MR')
plt.plot(np.arange(1,161),sig2, '-o' , label = 'periodic')
plt.yscale('log')
plt.ylabel('si')
plt.xlabel('N')
plt.legend()
plt.grid()

plt.subplot('312')
plt.plot(np.arange(1,160),1-(cumsum1[:-1]/cumsum1[-1]), '-o' , label = 'MR')
plt.plot(np.arange(1,160),1-(cumsum2[:-1]/cumsum2[-1]), '-o', label = 'perdiodic' )
plt.yscale('log')
plt.ylabel('eps POD')
plt.xlabel('N : Number of basis functions')
plt.grid()

plt.subplot('313')
plt.plot(np.arange(1,160),np.sqrt(1-(cumsum1[:-1]/cumsum1[-1])), '-o' , label = 'MR')
plt.plot(np.arange(1,160),np.sqrt(1-(cumsum2[:-1]/cumsum2[-1])), '-o', label = 'perdiodic' )
plt.yscale('log')
plt.ylabel('sqrt(eps POD)')
plt.xlabel('N : Number of basis functions')
plt.grid()

plt.show()
