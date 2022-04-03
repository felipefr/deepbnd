import os, sys

# import generation_deepBoundary_lib as gdb
import matplotlib.pyplot as plt
import numpy as np

# import myTensorflow as mytf
from timeit import default_timer as timer


from deepBND.__init__ import *
import deepBND.core.data_manipulation.wrapper_h5py as myhd

# import plotUtils

# Test Loading 

folderBasis = './'


# # ======================= Reading theoretical POD =================== 
nsTrain = 51200
nbasis = 160
nameEigen = folderBasis + 'Wbasis.hd5'


errorPOD = {}
for loadType in ['A', 'S']:
    eig = myhd.loadhd5(nameEigen, 'sig_%s'%loadType)   
    errorPOD[loadType] = np.zeros(nbasis)
    
    for i in range(nbasis):
        errorPOD[loadType][i] = np.sum(eig[i:])/nsTrain
    

# # ================ Alternative Prediction ========================================== 
folderImages = './'
suptitle = 'RB-POD Approximation Error' 
plt.figure(1,(5.0,3.5))
plt.title(suptitle.format('', ''))
plt.plot(np.arange(160),errorPOD['A'][:160],'--', label = 'Axial')
plt.plot(np.arange(160),errorPOD['S'][:160],'--', label = 'Shear')
# plt.ylim(1.0e-10,1.0e-1)
plt.yscale('log')
plt.xlabel('$N_{rb}$ (Number of RB)')
plt.ylabel('$\mathcal{E}_{POD}^2$ (Squared Error)')
plt.grid()
plt.legend(loc = 'best')
plt.tight_layout()
plt.savefig('POD_error.eps')
plt.savefig('POD_error.pdf')


