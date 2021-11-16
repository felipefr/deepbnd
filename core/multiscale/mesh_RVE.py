import sys, os
sys.path.insert(0,'../../')
import numpy as np
from core.mesh.ellipse2_mesh import ellipseMesh2
from core.mesh.ellipse_two_domains_mesh import ellipseMesh2Domains
import core.sampling.generation_inclusions as geni


class paramRVE_default:
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


def buildRVEmesh(paramRVEdata, nameMesh, isOrdinated = False, size = 'reduced'):

    p = paramRVE_default() # load default parameters
    
    if(isOrdinated): # it is already ordered (internal, after external)
        permTotal = np.arange(0,p.Nx*p.Ny).astype('int')
    else: # it is NOT ordered already ordered (internal, after external)
        permTotal = geni.orderedIndexesTotal(p.Nx,p.Ny,p.NxL)

    paramRVEdata[:,:] = paramRVEdata[permTotal,:]
    
    if(size == 'reduced'): # It was chosen 2x2 for the moment
        meshGMSH = ellipseMesh2(paramRVEdata[:4,:], p.x0L, p.y0L, p.LxL, p.LyL, p.lcar) 
        meshGMSH.setTransfiniteBoundary(p.NpLxL)
        
    elif(size == 'full'):
        meshGMSH = ellipseMesh2Domains(p.x0L, p.y0L, p.LxL, p.LyL, p.NL, paramRVEdata, 
                                            p.Lxt, p.Lyt, p.lcar, x0 = p.x0, y0 = p.y0)
        meshGMSH.setTransfiniteBoundary(p.NpLxt)
        meshGMSH.setTransfiniteInternalBoundary(p.NpLxL)   
            
    meshGMSH.write(nameMesh, opt = 'fenics')