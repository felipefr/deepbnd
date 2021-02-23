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
import myHDF5 as myhd
import dataManipMisc as dman 
from sklearn.preprocessing import MinMaxScaler
import miscPosprocessingTools as mpt
import myHDF5 as myHDF5

import json
import copy

import matplotlib.pyplot as plt
import symmetryLib as syml

# Test Loading 
folderTest = './models/dataset_testNew3/'
nameYtest = folderTest + 'Y_extended.h5'
nameEllipseDataTest = folderTest + 'ellipseData.h5'
nameSnapsTest = folderTest + 'snapshots.h5'

Ytest = myhd.loadhd5(nameYtest, 'Ylist')
ellipseDataTest = myhd.loadhd5(nameEllipseDataTest, 'ellipseData')
IsolT = myhd.loadhd5(nameSnapsTest,['solutions_trans'])[0]

ns0 = 0
ns1 = 1000

plt.figure(1,(13,7))
plt.suptitle('Test seed=19 / RB extended ({0}:{1})'.format(ns0,ns1))
plt.subplot('321')
plt.title('Histogram Y_1')
plt.hist(Ytest[ns0:ns1,0],bins = 20)

plt.subplot('322')
plt.title('Histogram Y_2')
plt.hist(Ytest[ns0:ns1,1], bins = 20)

plt.subplot('323')
plt.title('Histogram Y_3')
plt.hist(Ytest[ns0:ns1,2],bins = 20)

plt.subplot('324')
plt.title('Histogram Y_4')
plt.hist(Ytest[ns0:ns1,3], bins = 20)

plt.subplot('325')
plt.title('Histogram Y_5')
plt.hist(Ytest[ns0:ns1,4],bins = 20)

plt.subplot('326')
plt.title('Histogram Y_6')
plt.hist(Ytest[ns0:ns1,5], bins = 20)

plt.tight_layout()

plt.savefig("newTest3_extended.png")