import sys
sys.path.insert(0, '../../utils/')

from files import *
import numpy as np

# merge('data_newTest_new.hdf5', ['Test/Y'], np.arange(5).astype('int'))
# merge('solution_NewTest.hdf5', ['Test/sol'], np.arange(5).astype('int'), axis = 1)
# merge('posProc_newTest.hdf5', ['Test/vonMises'], np.arange(5).astype('int'))
# merge('paramFile.hdf5', ['Test/attrs/shift'], np.arange(5).astype('int'), mode = 'r+')

# merge(['data_newTest_new_{0}.hdf5'.format(i) for i in range(5)], 'data_newTest_new.hdf5', ['Test/X'], ['Test/XX'], mode = 'r+')

f = h5py.File('data_newTest_new.hdf5','a')

# X1 = f['Test']['X']
del f['Test']['XX']

# print(np.allclose(X1,X2))

