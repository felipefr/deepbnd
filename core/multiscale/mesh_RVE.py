import sys, os
import numpy as np
# from deepBND.core.mesh.ellipse2_mesh import ellipseMesh2
# from deepBND.core.mesh.ellipse_two_domains_mesh import ellipseMesh2Domains
import deepBND.creation_model.dataset.generation_inclusions as geni


from fetricks.fenics.mesh.ellipsoidal_inclusions_mesh import ellipsoidalInclusionsMesh as ellipseMesh2
from fetricks.fenics.mesh.ellipsoidal_inclusions_twodomains_mesh import ellipsoidalInclusionsTwoDomainsMesh as ellipseMesh2Domains


class paramRVE_default:
    def __init__(self, NxL = 2, NyL = 2, maxOffset = 2, Vfrac = 0.282743):
        self.Vfrac = Vfrac    
        self.maxOffset = maxOffset
        self.NxL = NxL
        self.NyL = NyL
        self.H = 1.0 # size of each square
        self.NL = self.NxL*self.NyL
        self.LxL = self.NxL*self.H
        self.LyL = self.NyL*self.H
        self.x0L = - 0.5*self.LxL
        self.y0L = - 0.5*self.LyL
        self.lcar = (2/30)*self.H # normally is 2/30
        self.Nx = (self.NxL+2*self.maxOffset)
        self.Ny = (self.NyL+2*self.maxOffset)
        self.Lxt = self.Nx*self.H
        self.Lyt = self.Ny*self.H
        self.NpLxt = int(self.Lxt/self.lcar) + 1
        self.NpLxL = int(self.LxL/self.lcar) + 1
        self.x0 = -0.5*self.Lxt
        self.y0 = -0.5*self.Lyt
        self.r0 = 0.2*self.H
        self.r1 = 0.4*self.H
        self.rm = self.H*np.sqrt(self.Vfrac/np.pi)


def buildRVEmesh(paramRVEdata, nameMesh, isOrdered = False, size = 'reduced', 
                 NxL = 2, NyL = 2, maxOffset = 2, Vfrac = 0.282743, lcar = None):

    p = paramRVE_default(NxL, NyL, maxOffset, Vfrac) # load default parameters
    p.lcar = lcar*p.H if type(lcar) is not type(None) else p.lcar
    
    
    if(isOrdered): # it is already ordered (internal, after external)
        permTotal = np.arange(0,p.Nx*p.Ny).astype('int')
    else: # it is NOT ordered already ordered (internal, after external)
        permTotal = geni.orderedIndexesTotal(p.Nx,p.Ny,p.NxL)

    paramRVEdata[:,:] = paramRVEdata[permTotal,:]
    
    if(size == 'reduced'):
        meshGMSH = ellipseMesh2(paramRVEdata[:p.NxL*p.NyL,:], p.x0L, p.y0L, p.LxL, p.LyL, p.lcar) 
        meshGMSH.setTransfiniteBoundary(p.NpLxL)
        
    elif(size == 'full'):
        meshGMSH = ellipseMesh2Domains(p.x0L, p.y0L, p.LxL, p.LyL, p.NL, paramRVEdata, 
                                            p.Lxt, p.Lyt, p.lcar, x0 = p.x0, y0 = p.y0)
        meshGMSH.setTransfiniteBoundary(p.NpLxt)
        meshGMSH.setTransfiniteInternalBoundary(p.NpLxL)   
            
    meshGMSH.write(nameMesh, opt = 'fenics')