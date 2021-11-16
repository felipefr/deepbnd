import sys, os
sys.path.insert(0,'../../')
import numpy as np
from core.mesh.ellipse2_mesh import ellipseMesh2
from core.mesh.ellipse_two_domains_mesh import ellipseMesh2Domains
import core.sampling.generation_inclusions as geni

def buildRVEmesh(paramRVEdata, nameMesh, isOrdenated = False, size = 'reduced'):

    p = geni.paramRVE() # load default parameters
    
    if(isOrdenated): # it is already ordered (internal, after external)
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