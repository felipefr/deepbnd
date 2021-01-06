import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0,'../')
sys.path.insert(0,'../../utils/')
import matplotlib.pyplot as plt
import numpy as np

import myTensorflow as mytf
from timeit import default_timer as timer

import h5py
import pickle
# import Generator as gene
import myHDF5 
import dataManipMisc as dman 
from sklearn.preprocessing import MinMaxScaler
import miscPosprocessingTools as mpt
import myHDF5 as myhd


historyArrayMaxmin = np.loadtxt( 'differentModels/history_maxmin_nX36_nY40_6.txt')
historyArrayP2 = np.loadtxt( 'differentModels/history_p2_nX36_nY40_5.txt')
historyArrayP3 = np.loadtxt( 'differentModels/history_p3_nX36_nY40_4.txt')
historyArrayP4 = np.loadtxt( 'differentModels/history_p4_nX36_nY40_3.txt')
historyArrayP4_newLoss = np.loadtxt( 'differentModels/history_p4_nX36_nY40_8.txt')


plt.figure(1)
plt.title('Training error')
plt.plot(historyArrayMaxmin[0,:] , label = 'p=1')
plt.plot(historyArrayP2[0,:] , label = 'p=2')
plt.plot(historyArrayP3[0,:] , label = 'p=3')
plt.plot(historyArrayP4[0,:] , label = 'p=4')
plt.legend()
plt.grid()
plt.yscale('log')
plt.ylabel('mse')
plt.xlabel('epochs')
plt.savefig('differentModels/trainingError_comparison.png')
plt.show()


plt.figure(2)
plt.title('Validation error')
plt.plot(historyArrayMaxmin[1,:] , label = 'p=1')
plt.plot(historyArrayP2[1,:] , label = 'p=2')
plt.plot(historyArrayP3[1,:] , label = 'p=3')
plt.plot(historyArrayP4[1,:] , label = 'p=4')
plt.legend()
plt.grid()
plt.yscale('log')
plt.ylabel('mse')
plt.xlabel('epochs')
plt.savefig('differentModels/validationError_comparison.png')
plt.show()


plt.figure(3)
plt.title('Training loss')
plt.plot(historyArrayP4[0,:] , label = 'p=4')
plt.plot(historyArrayP4_newLoss[0,:] , label = 'p=4 , weighted loss')
plt.legend()
plt.grid()
plt.yscale('log')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.savefig('differentModels/trainingLoss_comparison_newLoss.png')
plt.show()


plt.figure(4)
plt.title('Validation loss')
plt.plot(historyArrayP4[1,:] , label = 'p=4')
plt.plot(historyArrayP4_newLoss[1,:] , label = 'p=4 , weighted loss')
plt.legend()
plt.grid()
plt.yscale('log')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.savefig('differentModels/validationLoss_comparison_newLoss.png')
plt.show()


plt.figure(5)
plt.title('Training MSE')
plt.plot(historyArrayP4[2,:] , label = 'p=4')
plt.plot(historyArrayP4_newLoss[2,:] , label = 'p=4 , weighted loss')
plt.legend()
plt.grid()
plt.yscale('log')
plt.ylabel('mse')
plt.xlabel('epochs')
plt.savefig('differentModels/trainingMSE_comparison_newLoss.png')
plt.show()


plt.figure(6)
plt.title('Validation MSE')
plt.plot(historyArrayP4[3,:] , label = 'p=4')
plt.plot(historyArrayP4_newLoss[3,:] , label = 'p=4 , weighted loss')
plt.legend()
plt.grid()
plt.yscale('log')
plt.ylabel('mse')
plt.xlabel('epochs')
plt.savefig('differentModels/validationMSE_comparison_newLoss.png')
plt.show()



# print(end - start)

# plt.figure(1)
# plt.plot(hist.history['mse'] , label = 'train')
# plt.plot(hist.history['val_mse'], label = 'validation')
# plt.legend()
# plt.grid()
# plt.yscale('log')
# plt.ylabel('mse')
# plt.xlabel('epochs')
# plt.savefig(fnames['prefix_out'] + '/plot_mse_{0}.png'.format(run_id))


# plt.figure(2)
# plt.plot(hist.history['mae'] , label = 'train')
# plt.plot(hist.history['val_mae'], label = 'validation')
# plt.legend()
# plt.grid()
# plt.yscale('log')
# plt.ylabel('mae')
# plt.xlabel('epochs')
# plt.savefig(fnames['prefix_out'] + '/plot_mae_{0}.png'.format(run_id))


# plt.show()


# 1

