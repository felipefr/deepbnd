import os, sys

# import generation_deepBoundary_lib as gdb
import matplotlib.pyplot as plt
import numpy as np

# import myTensorflow as mytf
from timeit import default_timer as timer


from deepBND.__init__ import *
import deepBND.core.data_manipulation.wrapper_h5py as myhd

import fetricks.plotting.misc as plot

# Test Loading 
folderDataset = rootDataPath + "/review2_smaller/dataset_coarser/"

# # ======================= Reading theoretical POD ===================
suffix = "translation" 
ns = 60000
nbasis = 160
nameEigen = folderDataset + 'Wbasis_%s.hd5'%suffix


errorPOD = {}
for loadType in ['A', 'S']:
    eig = myhd.loadhd5(nameEigen, 'sig_%s'%loadType)   
    errorPOD[loadType] = np.zeros(nbasis)
    
    for i in range(nbasis):
        errorPOD[loadType][i] = np.sum(eig[i:])/ns
    

# # ================ Alternative Prediction ========================================== 
folderImages = './'
suptitle = 'RB-POD Approximation Error' 
plt.figure(1,(5.0,3.5))
plt.title(suptitle.format('', ''))
plt.plot(np.arange(nbasis), errorPOD['A'][:nbasis],'--', label = 'Axial')
plt.plot(np.arange(nbasis), errorPOD['S'][:nbasis],'--', label = 'Shear')
plt.yticks([np.power(1.0,-i) for i in np.arange(11,2,-1)])
plt.ylim(1.0e-11,1.0e-1)
plt.yscale('log')
plt.xlabel('$N_{rb}$ (Number of RB)')
plt.ylabel('$\mathcal{E}_{POD}^2$ (Squared Error)')

plt.grid()
plt.legend(loc = 'best')
plt.tight_layout()
plt.savefig('POD_error_%s.eps'%suffix)
plt.savefig('POD_error_%s.pdf'%suffix)


