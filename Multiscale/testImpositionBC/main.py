import sys, os
import numpy as np
sys.path.insert(0, '../../utils/')
from timeit import default_timer as timer
import ioFenicsWrappers as iofe
import Snapshots as snap
import Generator as gene
import genericParam as gpar
import fenicsWrapperElasticity as fela
import wrapperPygmsh as gmsh
# import pyvista as pv
import generationInclusions as geni
import ioFenicsWrappers as iofe
import fenicsMultiscale as fmts
import dolfin as df
import generatorMultiscale as gmts
import fenicsUtils as feut
import elasticity_utils as elut
import multiphenicsMultiscale as mpms


folder = "./"
radFile = folder + "test_{0}_{1}.{2}"

contrast = 10.0
E2 = 1.0
E1 = contrast*E2 # inclusions
nu1 = 0.3
nu2 = 0.3

mu1 = elut.eng2mu(nu1,E1)
lamb1 = elut.eng2lambPlane(nu1,E1)
mu2 = elut.eng2mu(nu2,E2)
lamb2 = elut.eng2lambPlane(nu2,E2)

param = np.array([[lamb1, mu1], [lamb2,mu2]])
        
eps = np.zeros((2,2))
eps[1,0] = 0.5
eps = 0.5*(eps + eps.T)

np.random.seed(10)

replications = 2

offset = 0
Lx = Ly = 1.0
ifPeriodic = False 
NxL = NyL = 2
x0L = y0L = offset*Lx/(1+2*offset)
LxL = LyL = Lx/(1+2*offset)
r0 = 0.2*LxL/NxL
r1 = 0.4*LxL/NxL

ellipseData = geni.circularRegular2Regions(r0, r1, NxL, NyL, Lx, Ly, offset, ordered = True)

times = 1

lcar = 0.1*Lx/(NxL*times)

meshGMSH = gmsh.ellipseMeshRepetition(times, ellipseData, Lx, Ly , lcar, ifPeriodic)
meshGMSH.setTransfiniteBoundary(int(Lx/lcar) + 1)

meshGeoFile = radFile.format(times,'','geo')
meshXmlFile = radFile.format(times,'','xml')
meshMshFile = radFile.format(times,'','msh')

meshGMSH.write(meshGeoFile,'geo')
os.system('gmsh -2 -algo del2d -format msh2 ' + meshGeoFile)

os.system('dolfin-convert {0} {1}'.format(meshMshFile, meshXmlFile))

meshFenics = fela.EnrichedMesh(meshXmlFile)
meshFenics.createFiniteSpace('V', 'u', 'CG', 1)
        
sigmaEps = fmts.getSigmaEps(param, meshFenics, eps)

# sol1 = fmts.solveMultiscale(param, meshFenics, eps, op = 'MR_mp', others = [[2]])
# u1 = sol1.split()[0] # MR

u1 = mpms.solveMultiscale(param, meshFenics, eps, op = 'MR', others = [[2]])[0]

sigma1 = fmts.homogenisation(u1, meshFenics, fmts.getSigma(param,meshFenics), [0,1], sigmaEps)        

g = gmts.displacementGeneratorBoundary(0.0,0.0,1.0,1.0, 21)
u1_bound = g(meshFenics,u1)

# sol2 = fmts.solveMultiscale(param, meshFenics, eps, op = 'BCdirich', others = [[2], u1_bound])
# u2 = sol2.split()[0]

u2 = mpms.solveMultiscale(param, meshFenics, eps, op = 'BCdirich_lag', others = [[2], u1_bound])[0]

sigma2 = fmts.homogenisation(u2, meshFenics, fmts.getSigma(param,meshFenics), [0,1], sigmaEps)

print(sigma1 - sigma2)
print(np.sqrt(df.assemble(df.dot(u1-u2,u1-u2)*meshFenics.dx))) 


# POD test
# n = len(u1_bound.flatten())
# Ubasis = np.eye(n)
# x_eval = g.x_eval
# alpha = u1_bound.flatten()


# sol3 = fmts.solveMultiscale(param, meshFenics, eps, op = 'POD', others = [Ubasis, x_eval, alpha])
# # u2 = sol2.split()[0]
# # sigma2 = fmts.homogenisation(u2, meshFenics, fmts.getSigma(param,meshFenics), [0,1], sigmaEps)
    
