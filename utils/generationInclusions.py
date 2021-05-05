import numpy as np
from skopt.space import Space
from skopt.sampler import Lhs, Sobol

def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]


class paramRVE:
    def __init__(self, NxL = 2, NyL = 2, maxOffset = 2, Vfrac = 0.282743):
        self.Vfrac = Vfrac    
        self.maxOffset = maxOffset
        self.NxL = NxL
        self.NyL = NyL
        self.H = 1.0 # size of each square
        self.NL = self.NxL*self.NyL
        self.x0L = self.y0L = -self.H 
        self.LxL = self.LyL = 2*self.H
        self.lcar = (2/30)*self.H
        self.Nx = (self.NxL+2*self.maxOffset)
        self.Ny = (self.NyL+2*self.maxOffset)
        self.Lxt = self.Nx*self.H
        self.Lyt = self.Ny*self.H
        self.NpLxt = int(self.Lxt/self.lcar) + 1
        self.NpLxL = int(self.LxL/self.lcar) + 1
        self.x0 = -self.Lxt/2.0
        self.y0 = -self.Lyt/2.0
        self.r0 = 0.2*self.H
        self.r1 = 0.4*self.H
        self.rm = self.H*np.sqrt(self.Vfrac/np.pi)


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


def getScikitoptSample_LHSbyLevels(NR,ns,p,indexes,r0,r1, seed, op = 'lhs'):
    M = len(indexes)
    N = int(ns/(p**M))
    
    space = Space(NR*[(r0, r1)])
    if(op == 'lhs'):
        sampler = Lhs(lhs_type="centered", criterion=None)
    elif(op == 'lhs_maxmin'):
        sampler = Lhs(criterion="maximin", iterations=20)
    elif(op == 'sobol'):
        sampler = Sobol()
    
    Rlist = []
    np.random.seed(seed)
    
        
    rlim = [r0 + i*(r1-r0)/p for i in range(p+1)]
    faclim = [(rlim[i+1]-rlim[i])/(r1-r0) for i in range(p)]
    
    for pi in range(p**M):
        pibin = numberToBase(pi,p) # supposing p = 2
        pibin = pibin + (4-len(pibin))*[0] # to complete 4 digits
        
        Rlist.append( np.array(sampler.generate(space.dimensions, N)) )
        
        for j in range(len(Rlist[-1][0,:])):
            if(j not in indexes):
                Rlist[-1][:,j] = Rlist[0][:,j]
        
        for j, jj in enumerate(indexes): #j = 0,1,2,..., jj = I_0,I_1,I_2,...
            k = pibin[j]
            Rlist[-1][:,jj] = rlim[k] + faclim[k]*( Rlist[-1][:,jj] - r0 )
            
    
        R = np.concatenate(Rlist,axis = 0)
    

    return R
 
    
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

    ellipseData = getEllipse_emptyRadius(Nx,Ny,Lx = 1.0, Ly=1.0, x0 = 0.0, y0 = 0.0)
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