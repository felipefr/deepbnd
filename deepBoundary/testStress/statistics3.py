import os, sys
sys.path.insert(0, '../../utils/')
import matplotlib.pyplot as plt
import numpy as np
# import Generator as gene
import myHDF5 as myhd

# Test Loading 
folderTest = './models/dataset_axial3/'
nameXYtest = folderTest + 'XY_extended_WbasisSimple.h5'

Ytest = myhd.loadhd5(nameXYtest, 'Y')
load_sign = myhd.loadhd5(nameXYtest, 'load_sign')


ns = 10240
par = 0
ns0 = par*ns
ns1 = (par+1)*ns

Ytest_loadSigned = np.einsum('ij,i->ij', Ytest, load_sign)
# Ytest_loadSigned = np.einsum('ij,i->ij', Ytest, np.ones(7*ns))

indexes = np.concatenate( tuple( [np.arange(i*ns,(i+1)*ns) for i in range(7)] ) )


plt.figure(1,(13,7))
plt.suptitle('Sign x Projections : Axial dataset (extended)/RB (simple basis)'.format(par))
plt.subplot('321')
plt.title('Histogram Y_1')
plt.hist(Ytest_loadSigned[indexes,0],bins = 20)

plt.subplot('322')
plt.title('Histogram Y_2')
plt.hist(Ytest_loadSigned[indexes,1], bins = 20)

plt.subplot('323')
plt.title('Histogram Y_3')
plt.hist(Ytest_loadSigned[indexes,2],bins = 20)

plt.subplot('324')
plt.title('Histogram Y_4')
plt.hist(Ytest_loadSigned[indexes,3], bins = 20)

plt.subplot('325')
plt.title('Histogram Y_5')
plt.hist(Ytest_loadSigned[indexes,4],bins = 20)

plt.subplot('326')
plt.title('Histogram Y_6')
plt.hist(Ytest_loadSigned[indexes,5], bins = 20)

# plt.tight_layout()

# plt.savefig("Projections_Shear_WbasisExtendedOld.png")
plt.show()

