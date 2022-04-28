import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np
import deepBND.core.data_manipulation.utils as dman

import deepBND.core.data_manipulation.wrapper_h5py as myhd

def exportScale(filenameIn, filenameOut, nX, nY, Ylabel = 'Y', scalerType = "MinMax"):
    scalerX, scalerY = getDatasetsXY(nX, nY, filenameIn, Ylabel = Ylabel, scalerType = scalerType)[2:4]
    scalerLimits = np.zeros((max(nX,nY),4))
    scalerLimits[:nX,0:2] = scalerX.export_scale()
    scalerLimits[:nY,2:4] = scalerY.export_scale()

    np.savetxt(filenameOut, scalerLimits)

def importScale(filenameIn, nX, nY):
    scalerX = myMinMaxScaler()
    scalerY = myMinMaxScaler()
    scalerX.load_param(np.loadtxt(filenameIn)[:,0:2])
    scalerY.load_param(np.loadtxt(filenameIn)[:,2:4])
    scalerX.set_n(nX)
    scalerY.set_n(nY)
    
    return scalerX, scalerY


class myScaler:
    def __init__(self):
        self.n = 0
        
    def set_n(self,n):
        pass
    
    def load_param(self, param):
        pass 
    
    def export_scale(self):
        pass 
           
    def fit(self,x):
        pass
    
    def scaler(self, x, i):
        pass 
    
    def inv_scaler(self, x, i):
        pass 
    
    def transform(self,x):
        return np.array( [self.scaler(x[:,i],i) for i in range(self.n)] ).T
            
    def inverse_transform(self,x):
        return np.array( [self.inv_scaler(x[:,i],i) for i in range(self.n)] ).T


class myMinMaxScaler(myScaler):
    def __init__(self):
        self.data_min_ = []
        self.data_max_ = []
        super().__init__()
                
    def set_n(self,n):
        self.n = n

        self.data_min_ = self.data_min_[:self.n]
        self.data_max_ = self.data_max_[:self.n]

    def scaler(self, x, i):
        return (x - self.data_min_[i])/(self.data_max_[i]-self.data_min_[i])
    
    def inv_scaler(self, x, i):
        return (self.data_max_[i]-self.data_min_[i])*x + self.data_min_[i]

    def load_param(self, param):
        self.n = param.shape[0]
                
        self.data_min_ = param[:,0]
        self.data_max_ = param[:,1]
        
    def export_scale(self):
        return np.stack( (self.data_min_, self.data_max_ )).T
            
    def fit(self,x):
        self.n = x.shape[1]
        
        for i in range(self.n):
            self.data_min_.append(x[:,i].min())
            self.data_max_.append(x[:,i].max())
        
        self.data_min_ = np.array(self.data_min_)
        self.data_max_ = np.array(self.data_max_)


class myNormalisationScaler(myScaler):
    def __init__(self):
        self.data_mean = []
        self.data_std = []
        super().__init__()
        
    def set_n(self,n):
        self.n = n
        self.data_mean = self.data_mean[:self.n]
        self.data_std = self.data_std[:self.n]

    def scaler(self, x, i): 
        return (x - self.data_mean[i])/self.data_std[i]
    
    def inv_scaler(self, x, i):
        return (x*self.data_std[i] + self.data_mean[i])

    def load_param(self, param):
        self.n = param.shape[0]
                
        self.data_mean = param[:,0]
        self.data_std = param[:,1]
        
    def export_scale(self):
        return np.stack( (self.data_mean, self.data_std)).T
                             
    def fit(self,x):
        self.n = x.shape[1]
        
        for i in range(self.n):
            self.data_mean.append(np.mean(x[:,i]))
            self.data_std.append(np.std(x[:,i]))
        
        self.data_mean = np.array(self.data_mean)
        self.data_std = np.array(self.data_std)


    def transform(self,x):
        return np.array( [self.scaler(x[:,i],i) for i in range(self.n)] ).T
            
    def inverse_transform(self,x):
        return np.array( [self.inv_scaler(x[:,i],i) for i in range(self.n)] ).T


def writeDict(d):
    f = open(d['files']['net_settings'],'w')
    
    for keys, value in zip(d.keys(),d.values()):
        f.write("{0}: {1}\n".format(keys,value))
        
    f.close()
        
def getDatasetsXY(nX, nY, XYdatafile, scalerX = None, scalerY = None, Ylabel = 'Y', scalerType = 'MinMax'):
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
        scalerX = {'MinMax': myMinMaxScaler() , 'Normalisation': myNormalisationScaler()}[scalerType]            
        scalerX.fit(X)
    
    if(type(scalerY) == type(None)):
        scalerY = {'MinMax': myMinMaxScaler() , 'Normalisation': myNormalisationScaler()}[scalerType]            
        scalerY.fit(Y)
            
    return scalerX.transform(X), scalerY.transform(Y), scalerX, scalerY


# def getTraining(ns_start, ns_end, nX, nY, Xdatafile, Ydatafile, scalerX = None, scalerY = None):
#     X = np.zeros((ns_end - ns_start,nX))
#     Y = np.zeros((ns_end - ns_start,nY))
    
#     X = myhd.loadhd5(Xdatafile, 'ellipseData')[ns_start:ns_end,:nX,2]
#     Y = myhd.loadhd5(Ydatafile, 'Ylist')[ns_start:ns_end,:nY]

#     if(type(scalerX) == type(None)):
#         scalerX = MinMaxScaler()
#         scalerX.fit(X)
    
#     if(type(scalerY) == type(None)):
#         scalerY = MinMaxScaler()
#         scalerY.fit(Y)
            
#     return scalerX.transform(X), scalerY.transform(Y), scalerX, scalerY