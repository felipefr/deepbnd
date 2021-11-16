import numpy as np
from skopt.space import Space
from skopt.sampler import Lhs, Sobol

def getScikitoptSample(NR,ns,r0,r1, seed, op = 'lhs'):
    space = Space(NR*[(r0, r1)])
    if(op == 'lhs'):
        sampler = Lhs(lhs_type="centered", criterion=None)
    elif(op == 'lhs_maxmin'):
        sampler = Lhs(criterion="maximin", iterations=100)
    elif(op == 'sobol'):
        sampler = Sobol()
    
    np.random.seed(seed)
    
    return np.array(sampler.generate(space.dimensions, ns))

def getEllipse_emptyRadius(Nx,Ny,Lx = 1.0, Ly=1.0, x0 = 0.0, y0 = 0.0):
    N = Nx*Ny
    angle = 0.0
    e = 1.0
    
    hx = Lx/Nx
    hy = Ly/Ny
    
    cx = np.linspace(x0 + 0.5*hx, x0 + Lx - 0.5*hx, Nx)
    cy = np.linspace(y0 + 0.5*hy, y0 + Ly - 0.5*hy, Ny)
    cx,cy = np.meshgrid(cx,cy)
    
    cx = cx.flatten()
    cy = cy.flatten()

    return np.array([[cx[i],cy[i],0.0,e,angle] for i in range(N)])


def getRadiusExponential(r0, r1, theta = None, N = 0): # N is only used if theta is not supplied
    mu = 0.5*np.log(r1*r0)
    sig = 0.5*np.log(r1/r0)
    
    if(type(theta) == type(None)):
        theta = 2.0*np.random.rand(N) - 1.0
    
    r = np.exp(mu + sig*theta)

    return r 

def circularRegular(r0, r1, Nx,Ny,Lx = 1.0, Ly=1.0, x0 = 0.0, y0 = 0.0, theta = None):

    ellipseData = getEllipse_emptyRadius(Nx,Ny,Lx, Ly, x0, y0)
    N = len(ellipseData)
    
    ellipseData[:,2] = getRadiusExponential(r0, r1, theta, N)
        
    return ellipseData


# Returns the permutation for to turn the standard numeration (per rows and columns) to the internal to external
# layers numeration in just the internal box
def orderedIndexesBox(Nx,Ny,offset):
    indexes = np.arange(0,Nx*Ny).reshape((Nx,Ny))
    indexesL = indexes[offset : Nx - offset, offset : Ny - offset].flatten()
    
    return np.array(list(indexesL) + list( set(indexes.flatten()) - set(indexesL)), dtype = 'int')

def orderListBox(L):
    Nx = int(np.sqrt(float(len(L)))) 
    ind  = orderedIndexesBox(Nx,Nx,1)
    return list(np.array(L,dtype = 'int')[ind]) 


# Returns the permutation for to turn the standard numeration (per rows and columns) to the internal to external
# layers numeration 
def orderedIndexesTotal(Nx,Ny,minNx):
    L = list(np.arange(0,Nx*Ny).reshape((Nx,Ny)).astype('int').flatten())
    
    maxOffset = int((Nx - minNx)/2)
    NI = [ (Nx - 2*i)*(Ny - 2*i) for i in range(maxOffset)]
    
    for ni in NI:
        L = orderListBox(L[:ni]) + L[ni:]
                
    return np.array(L,dtype = 'int')


def inverseOrderedIndexesTotal(Nx,Ny,minNx):
    return np.argsort(orderedIndexesTotal(Nx,Ny,minNx))

             
def circularRegular2Regions(r0, r1, NxL, NyL, Lx = 1.0, Ly=1.0, offset = 0, ordered = False, x0 = 0.0, y0 = 0.0):
    Nx = NxL + 2*offset
    Ny = NyL + 2*offset
    
    paramExport = circularRegular(r0, r1, Nx,Ny,Lx, Ly, x0, y0)
    
    if(ordered):
        # return paramExport[orderedIndexesBox(Nx,Ny,offset)]
        return paramExport[orderedIndexesTotal(Nx,Ny,NxL)]        
    else:
        return paramExport, orderedIndexesTotal(Nx,Ny,NxL), orderedIndexesBox(Nx,Ny,offset)
    

