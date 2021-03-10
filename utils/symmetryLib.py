import os, sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler #
import myHDF5 as myhd ##
import copy 

# order 0 : matrix order
# order 1 : internal to external layer order (matrix convention ortherwise)
from generationInclusions import orderedIndexesTotal as order0_2_order1 # Nx,Ny,NL (NLx = NLy) 
from generationInclusions import inverseOrderedIndexesTotal as order1_2_order0 # Nx,Ny,NL (NLx = NLy)

# import dolfin as df

# T_MH = df.Expression(('-x[0]','x[1]'), degree = 1)
# T_MV = df.Expression(('x[0]','-x[1]'), degree = 1)
# T_MD = df.Expression(('-x[0]','-x[1]'), degree = 1)

T_halfpi = lambda pairs: [(j, int(np.sqrt(len(pairs))) - 1 - i ) for i,j in pairs ]
T_pi = lambda pairs: T_halfpi(T_halfpi(pairs))
T_mhalfpi = lambda pairs: T_halfpi(T_pi(pairs))
T_horiz = lambda pairs: [(int(np.sqrt(len(pairs))) - 1 - i, j ) for i,j in pairs ]
T_vert = lambda pairs: [(i , int(np.sqrt(len(pairs))) - 1 - j ) for i,j in pairs ]
T_diag = lambda pairs: T_horiz(T_vert(pairs))
pairs2ind = lambda pairs: [i + j*int(np.sqrt(len(pairs))) for i,j in pairs]
ind2pairs = lambda ind: [ ( k%int(np.sqrt(len(ind))), int(k/np.sqrt(len(ind))) )  for k in ind]

# inverse transformation of pairs : Tinv(T(p)) = p and T(Tinv(p)) = p. 
# This is done by: Transformin pairs -> convert pairs to indexes -> find the permutation to reorder indexes -> retransform indexes to pairs to pairs
inverse_T = lambda pairs, T: ind2pairs( np.argsort( pairs2ind(T(pairs)) ) ) 


# T_with_order1 : 1-ordered-vector --> 1-ordered-vector, given a transformation.
def perm_with_order1(T, N, NL) :
    Nx = Ny = int(np.sqrt(N))
    
    order0_2_order1_ = order0_2_order1(Nx,Ny,NL)
    order1_2_order0_ = order1_2_order0(Nx,Ny,NL)
    permTransform = np.array( pairs2ind(T(ind2pairs(np.arange(N)))) )

    return order1_2_order0_[permTransform[order0_2_order1_]]

# T_with_order0 : 0-ordered-vector --> 0-ordered-vector, given a transformation. 
def perm_with_order0(T, N) :    
    permTransform = np.array( pairs2ind(T(ind2pairs(np.arange(N)))) )

    return permTransform

def mirror(X, perm):
    perm = list(np.array(perm) - 1)
    X2 = copy.deepcopy(X)
    X2[:] = X2[perm]
    return X2

def mirrorHorizontal(X):
    perm = [2,1,4,3,8,7,6,5,10,9,12,11,16,15,14,13,22,21,20,19,18,17,24,23,26,25,28,27,30,29,36,35,34,33,32,31]
    return mirror(X,perm)

def mirrorVertical(X):
    perm = [3,4,1,2,13,14,15,16,11,12,9,10,5,6,7,8,31,32,33,34,35,36,29,30,27,28,25,26,23,24,17,18,19,20,21,22]
    return mirror(X,perm)

def mirrorDiagonal(X):
    return mirrorVertical(mirrorHorizontal(X))
    
def createSymmetricEllipseData(ellipseFileName, ellipseFileName_new = '', useFiles = True):
    if(useFiles):
        EllipseData = myhd.loadhd5(ellipseFileName, 'ellipseData')
    else:
        EllipseData = ellipseFileName # the name is the data itself.
        
    X = EllipseData[:,:,2]
    
    ns0 = len(X)
    nX = len(X[0])
    
    EllipseData_list = [EllipseData] 
    for mirror_func in [mirrorHorizontal, mirrorVertical, mirrorDiagonal]: 
        EllipseData_list.append(copy.deepcopy(EllipseData))
        for i in range(ns0):
            EllipseData_list[-1][i,:,2] = mirror_func(X[i,:])
    
    EllipseData_new = np.concatenate(tuple(EllipseData_list))
    
    if(useFiles):
        myhd.savehd5(ellipseFileName_new, EllipseData_new, 'ellipseData', mode = 'w' )

    return EllipseData_new

def getTraining_usingSymmetry(ns_start, ns_end, nX, nY, Xdatafile, Ydatafile, scalerX = None, scalerY = None):
    X = np.zeros((ns_end - ns_start,nX))
    Y = np.zeros((ns_end - ns_start,nY))
    
    X = myhd.loadhd5(Xdatafile, 'ellipseData')[ns_start:ns_end,:nX,2]
    Y = myhd.loadhd5(Ydatafile, 'Ylist')[ns_start:ns_end,:nY]
    
    XT1 = np.zeros((ns_end - ns_start,nX))
    XT2 = np.zeros((ns_end - ns_start,nX))
    XT3 = np.zeros((ns_end - ns_start,nX))

    for i in range(ns_end - ns_start):
        XT1[i,:] = mirrorHorizontal(X[i,:])
        XT2[i,:] = mirrorVertical(X[i,:])
        XT3[i,:] = mirrorDiagonal(X[i,:])
    
    YdatafileT1 = Ydatafile[:-3] + '_T1' + Ydatafile[-3:]
    YdatafileT2 = Ydatafile[:-3] + '_T2' + Ydatafile[-3:]
    YdatafileT3 = Ydatafile[:-3] + '_T3' + Ydatafile[-3:]
    
    print(YdatafileT1)
    YT1 = myhd.loadhd5(YdatafileT1, 'Ylist')[ns_start:ns_end,:nY]
    YT2 = myhd.loadhd5(YdatafileT2, 'Ylist')[ns_start:ns_end,:nY]
    YT3 = myhd.loadhd5(YdatafileT3, 'Ylist')[ns_start:ns_end,:nY]
    
    X = np.concatenate((X,XT1,XT2,XT3))
    Y = np.concatenate((Y,YT1,YT2,YT3))
    
    if(type(scalerX) == type(None)):
        scalerX = MinMaxScaler()
        scalerX.fit(X)
    
    if(type(scalerY) == type(None)):
        scalerY = MinMaxScaler()
        scalerY.fit(Y)
            
    return scalerX.transform(X), scalerY.transform(Y), scalerX, scalerY

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


def getDatasets(nX, nY, Xdatafile, Ydatafile, scalerX = None, scalerY = None):
    Xlist = []
    Ylist = []
    
    if(type(Xdatafile) != type([])):
        Xdatafile = [Xdatafile]
        Ydatafile = [Ydatafile]
    
    for Xdatafile_i, Ydatafile_i in zip(Xdatafile,Ydatafile):    
        Xlist.append(myhd.loadhd5(Xdatafile_i, 'ellipseData')[:,:nX,2])
        Ylist.append(myhd.loadhd5(Ydatafile_i, 'Ylist')[:,:nY])
    
    X = np.concatenate(tuple(Xlist),axis = 0)
    Y = np.concatenate(tuple(Ylist),axis = 0)
    
    if(type(scalerX) == type(None)):
        scalerX = MinMaxScaler()
        scalerX.fit(X)
    
    if(type(scalerY) == type(None)):
        scalerY = MinMaxScaler()
        scalerY.fit(Y)
            
    return scalerX.transform(X), scalerY.transform(Y), scalerX, scalerY

def getDatasetsXY(nX, nY, XYdatafile, scalerX = None, scalerY = None):
    Xlist = []
    Ylist = []
    
    if(type(XYdatafile) != type([])):
        XYdatafile = [XYdatafile]
    
    for XYdatafile_i in XYdatafile:    
        Xlist.append(myhd.loadhd5(XYdatafile_i, 'X')[:,:nX])
        Ylist.append(myhd.loadhd5(XYdatafile_i, 'Y')[:,:nY])
    
    X = np.concatenate(tuple(Xlist),axis = 0)
    Y = np.concatenate(tuple(Ylist),axis = 0)
    
    if(type(scalerX) == type(None)):
        scalerX = MinMaxScaler()
        scalerX.fit(X)
    
    if(type(scalerY) == type(None)):
        scalerY = MinMaxScaler()
        scalerY.fit(Y)
            
    return scalerX.transform(X), scalerY.transform(Y), scalerX, scalerY

