import numpy as np

def circularRegular(r0, r1, Nx,Ny,Lx = 1.0, Ly=1.0):
    N = Nx*Ny
    mu = 0.5*np.log(r1*r0)
    sig = 0.5*np.log(r1/r0)
    
    theta = 2.0*np.random.rand(N) - 1.0
    r = np.exp(mu + sig*theta)
        
    angle = 0.0
    e = 1.0
    
    hx = Lx/Nx
    hy = Ly/Ny
    
    cx = np.linspace(0.5*hx,Lx - 0.5*hx, Nx)
    cy = np.linspace(0.5*hy,Ly - 0.5*hy, Ny)
    cx,cy = np.meshgrid(cx,cy)
    
    cx = cx.flatten()
    cy = cy.flatten()

    return np.array([[cx[i],cy[i],r[i],e,angle] for i in range(N)])

def orderedIndexesBox(Nx,Ny,offset):
    indexes = np.arange(0,Nx*Ny).reshape((Nx,Ny))
    indexesL = indexes[offset : Nx - offset, offset : Ny - offset].flatten()
    
    return np.array(list(indexesL) + list( set(indexes.flatten()) - set(indexesL))) 
             
def circularRegular2Regions(r0, r1, NxL, NyL, Lx = 1.0, Ly=1.0, offset = 0, ordered = False):
    Nx = NxL + 2*offset
    Ny = NyL + 2*offset
    
    paramExport = circularRegular(r0, r1, Nx,Ny,Lx, Ly)
    
    if(ordered):
        return paramExport[orderedIndexesBox(Nx,Ny,offset)]        
    else:
        return paramExport