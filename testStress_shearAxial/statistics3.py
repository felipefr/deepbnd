import os, sys
sys.path.insert(0, '../utils/')
import matplotlib.pyplot as plt
import numpy as np
# import Generator as gene
import myHDF5 as myhd

# Test Loading 
folderTest = './models/dataset_axial2/'
nameXYtest = folderTest + 'XY_Wbasis3_extended.h5'

Ytest = myhd.loadhd5(nameXYtest, 'Y')
# load_sign = myhd.loadhd5(nameXYtest, 'load_sign')


ns = 5120
par = 0
ns0 = par*ns
ns1 = (par+1)*ns

# Ytest_loadSigned = np.einsum('ij,i->ij', Ytest, load_sign)
# Ytest_loadSigned = np.einsum('ij,i->ij', Ytest, np.ones(7*ns))

# indexes = np.concatenate( tuple( [np.arange(i*ns,(i+1)*ns) for i in range(7)] ) )
indexes = np.arange(ns)


plt.figure(1,(13,7))
plt.suptitle('Projections : Axial dataset val/RB (extended)'.format(par))
plt.subplot('321')
plt.title('Histogram Y_1')
plt.hist(Ytest[indexes,0],bins = 20)

plt.subplot('322')
plt.title('Histogram Y_2')
plt.hist(Ytest[indexes,1], bins = 20)

plt.subplot('323')
plt.title('Histogram Y_3')
plt.hist(Ytest[indexes,2],bins = 20)

plt.subplot('324')
plt.title('Histogram Y_4')
plt.hist(Ytest[indexes,3], bins = 20)

plt.subplot('325')
plt.title('Histogram Y_5')
plt.hist(Ytest[indexes,4],bins = 20)

plt.subplot('326')
plt.title('Histogram Y_6')
plt.hist(Ytest[indexes,5], bins = 20)

plt.tight_layout()
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9)

plt.savefig("Projections_dataset_axial2_extended.png")
plt.show()

