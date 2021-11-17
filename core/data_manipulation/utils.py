import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np

import deepBND.core.data_manipulation.wrapper_h5py as myhd

def exportScale(filenameIn, filenameOut, nX, nY, Ylabel = 'Y'):
    scalerX, scalerY = getDatasetsXY(nX, nY, filenameIn, Ylabel = Ylabel)[2:4]
    scalerLimits = np.zeros((max(nX,nY),4))
    scalerLimits[:nX,0] = scalerX.data_min_
    scalerLimits[:nX,1] = scalerX.data_max_
    scalerLimits[:nY,2] = scalerY.data_min_
    scalerLimits[:nY,3] = scalerY.data_max_

    np.savetxt(filenameOut, scalerLimits)

def importScale(filenameIn, nX, nY):
    scalerX = myMinMaxScaler()
    scalerY = myMinMaxScaler()
    scalerX.fit_limits(np.loadtxt(filenameIn)[:,0:2])
    scalerY.fit_limits(np.loadtxt(filenameIn)[:,2:4])
    scalerX.set_n(nX)
    scalerY.set_n(nY)
    
    return scalerX, scalerY

class myMinMaxScaler:
    def __init__(self):
        self.data_min_ = []
        self.data_max_ = []
        self.n = 0
        
    def set_n(self,n):
        self.n = n

        self.data_min_ = self.data_min_[:self.n]
        self.data_max_ = self.data_max_[:self.n]

    
    def fit_limits(self,limits):
        self.n = limits.shape[0]
                
        self.data_min_ = limits[:,0]
        self.data_max_ = limits[:,1]

        self.scaler = lambda x,i : (x - self.data_min_[i])/(self.data_max_[i]-self.data_min_[i])
        self.inv_scaler = lambda x,i : (self.data_max_[i]-self.data_min_[i])*x + self.data_min_[i]
                
    def fit(self,x):
        self.n = x.shape[1]
        
        for i in range(self.n):
            self.data_min_.append(x[:,i].min())
            self.data_max_.append(x[:,i].max())
        
        self.data_min_ = np.array(self.data_min_)
        self.data_max_ = np.array(self.data_max_)

        self.scaler = lambda x,i : (x - self.data_min_[i])/(self.data_max_[i]-self.data_min_[i])
        self.inv_scaler = lambda x,i : (self.data_max_[i]-self.data_min_[i])*x + self.data_min_[i]


    def transform(self,x):
        return np.array( [self.scaler(x[:,i],i) for i in range(self.n)] ).T
            
    def inverse_transform(self,x):
        return np.array( [self.inv_scaler(x[:,i],i) for i in range(self.n)] ).T


def writeDict(d):
    f = open(d['file_net'],'w')
    
    for keys, value in zip(d.keys(),d.values()):
        f.write("{0}: {1}\n".format(keys,value))
        
    f.close()
    
    
def getTraining(ns_start, ns_end, nX, nY, Xdatafile, Ydatafile, scalerX = None, scalerY = None):
    X = np.zeros((ns_end - ns_start,nX))
    Y = np.zeros((ns_end - ns_start,nY))
    
    X = myhd.loadhd5(Xdatafile, 'ellipseData')[ns_start:ns_end,:nX,2]
    Y = myhd.loadhd5(Ydatafile, 'Ylist')[ns_start:ns_end,:nY]

    if(type(scalerX) == type(None)):
        scalerX = MinMaxScaler()
        scalerX.fit(X)
    
    if(type(scalerY) == type(None)):
        scalerY = MinMaxScaler()
        scalerY.fit(Y)
            
    return scalerX.transform(X), scalerY.transform(Y), scalerX, scalerY

def getDatasetsXY(nX, nY, XYdatafile, scalerX = None, scalerY = None, Ylabel = 'Y'):
    Xlist = []
    Ylist = []
    
    if(type(XYdatafile) != type([])):
        XYdatafile = [XYdatafile]
    
    for XYdatafile_i in XYdatafile:    
        Xlist.append(myhd.loadhd5(XYdatafile_i, 'X')[:,:nX])
        Ylist.append(myhd.loadhd5(XYdatafile_i, Ylabel)[:,:nY])
    
    X = np.concatenate(tuple(Xlist),axis = 0)
    Y = np.concatenate(tuple(Ylist),axis = 0)
    
    if(type(scalerX) == type(None)):
        scalerX = myMinMaxScaler()
        scalerX.fit(X)
    
    if(type(scalerY) == type(None)):
        scalerY = myMinMaxScaler()
        scalerY.fit(Y)
            
    return scalerX.transform(X), scalerY.transform(Y), scalerX, scalerY
