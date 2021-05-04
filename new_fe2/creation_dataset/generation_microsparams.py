import sys, os
sys.path.insert(0, '../../utils/')

import matplotlib.pyplot as plt
import numpy as np
import generationInclusions as geni
import myHDF5 as myhd

f = open("../../../rootDataPath.txt")
rootData = f.read()[:-1]
f.close()

folder = rootData + "/new_fe2/dataset/"

p = geni.paramRVE()
NR = p.Nx*p.Ny # 36

ns = 10240 # test

# Radius Generation
seed = 17 # for the test   
np.random.seed(seed)

os.system('rm ' + folder +  'paramRVEdataset_validation.hd5')
X, f = myhd.zeros_openFile(filename = folder +  'paramRVEdataset_validation.hd5',  shape = (ns,NR,5), label = 'param', mode = 'w-')

ellipseData_pattern = geni.getEllipse_emptyRadius(p.Nx,p.Ny,p.Lxt, p.Lyt, p.x0, p.y0)

thetas = geni.getScikitoptSample(NR,ns, -1.0, 1.0,  seed, op = 'lhs')

for i in range(ns):
    print("inserting on ", i)
    X[i,:,:] =  ellipseData_pattern 
    X[i,:,2] = geni.getRadiusExponential(p.r0, p.r1, thetas[i,:])


f.close()



