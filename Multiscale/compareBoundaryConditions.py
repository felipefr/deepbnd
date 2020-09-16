import numpy as np
import sys
sys.path.insert(0, '../utils/')
from timeit import default_timer as timer
import elasticity_utils as elut
import ioFenicsWrappers as iofe
import Snapshots as snap
import sys, os
sys.path.insert(0, '../utils/')
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

Lx = Ly = 1.0
lcar = 0.02
ifPeriodic = False 
NxL = NyL = 2
offset = 1
x0L = y0L = 0.25
LxL = LyL = 0.5
r0 = 0.3*LxL/NxL
r1 = 0.4*LxL/NxL

np.random.seed(10)

ellipseData = geni.circularRegular2Regions(r0, r1, NxL, NyL, Lx, Ly, offset, ordered = True)
meshGMSH =  gmsh.ellipseMesh2DomainsPhysicalMeaning(x0L, y0L, LxL, LyL, NxL*NyL, ellipseData , Lx, Ly, lcar, ifPeriodic) 


meshGMSH.write('test3.geo','geo')
os.system('gmsh -2 -algo del2d -format msh2 test3.geo')

os.system("dolfin-convert test3.msh test7.xml")

meshFenics = fela.EnrichedMesh('test7.xml')
meshFenics.createFiniteSpace('V', 'u', 'CG', 1)
        
param = np.array([ [10.0,15.0], [20.0,30.0], [10.0,15.0], [20.0,30.0] ])

eps = np.zeros((2,2))
eps[0,0] = 0.1

sigmaEps = fmts.getSigmaEps(param, meshFenics, eps)
vol = df.assemble(df.Constant(1.0)*meshFenics.dx)

# Traditional way: mixed MR multiscale
u_trad, p, P = fmts.solveMultiscaleMR(param, meshFenics, eps)

sigmaL_trad = fmts.homogenisation_noMesh(u_trad, fmts.getSigma(param,meshFenics), [0,1], sigmaEps)
sigma_trad = fmts.homogenisation_noMesh(u_trad, fmts.getSigma(param,meshFenics), [0,1,2,3], sigmaEps)

sigma_LM = feut.Integral(P, meshFenics.dx, shape = (2,2))/vol

print(sigmaL_trad) 
print(sigma_trad)
print(sigma_LM)

iofe.postProcessing_complete(u_trad, "solution_trad.xdmf", ['u', 'lame', 'vonMises'], param , rename = False)

# Traditional way: mixed MR multiscale (all Split)
# u, p, P = fmts.solveMultiscaleMR_allSplit(param, meshFenics, eps)

# sigmaL_trad = fmts.homogenisation_allSplit(u, fmts.getSigma(param,meshFenics), [0,1], sigmaEps)
# sigma_trad = fmts.homogenisation_allSplit(u,  fmts.getSigma(param,meshFenics), [0,1,2,3], sigmaEps)

# sigma_LM = feut.Integral(P, meshFenics.dx, shape = (2,2))/vol

# print(sigmaL_trad) 
# print(sigma_trad)
# print(sigma_LM)


# Via dirichlet boundary
g = gmts.displacementGeneratorBoundary(0.0,0.0,1.0,1.0,800)
uBound = g(meshFenics,u_trad)  
uDirichlet = fmts.PointExpression(uBound, g)
u_diri = fmts.solveMultiscale_pointDirichlet(param, meshFenics, eps, uDirichlet)

sigmaL_diri = fmts.homogenisation_noMesh(u_diri, fmts.getSigma(param,meshFenics), [0,1], sigmaEps)
sigma_diri = fmts.homogenisation_noMesh(u_diri, fmts.getSigma(param,meshFenics), [0,1,2,3], sigmaEps)

print(sigmaL_diri)
print(sigma_diri)

iofe.postProcessing_complete(u_diri, "solution_diri.xdmf", ['u', 'lame', 'vonMises'], param)

# Comparison norms
print(np.linalg.norm(sigmaL_diri - sigmaL_trad))
print(np.linalg.norm(sigma_diri - sigma_trad))
print(np.sqrt(df.assemble(df.dot(u_diri-u_trad,u_diri-u_trad)*meshFenics.dx)))


# Via dirichlet boundary (middle)
volL = df.assemble(df.Constant(1.0)*meshFenics.dx(0)) + df.assemble(df.Constant(1.0)*meshFenics.dx(1))
epsL = (feut.Integral(fela.epsilon(u_trad), meshFenics.dx(0), shape = (2,2)) + \
        feut.Integral(fela.epsilon(u_trad), meshFenics.dx(1), shape = (2,2)))/volL + eps

sigmaEpsL = fmts.getSigmaEps(param, meshFenics, epsL)

g = gmts.displacementGeneratorBoundary(x0L,y0L,LxL,LyL,200)
uBound = g(meshFenics,u_trad)  
uBoundL = np.zeros(uBound.shape)
y = g.x_eval
for i in range(y.shape[0]):
    dy = np.dot(epsL-eps,y[i,:])
    uBoundL[2*i:2*(i+1),0] = uBound[2*i:2*(i+1),0] + dy

uDirichletL = fmts.PointExpression(uBoundL, g)
u_diriL = fmts.solveMultiscale_pointDirichlet(param, meshFenics, epsL, uDirichletL,[5])

sigmaL_diri2 = fmts.homogenisation_noMesh(u_diriL, fmts.getSigma(param,meshFenics), [0,1], sigmaEpsL)

print(sigmaL_diri2)

iofe.postProcessing_complete(u_diriL, "solution_diriL.xdmf", ['u', 'lame', 'vonMises'], param)

# Comparison norms
print(np.linalg.norm(sigmaL_diri2 - sigmaL_diri))
print(np.sqrt(df.assemble(df.dot(u_diri-u_trad,u_diri-u_trad)*meshFenics.dx(0)) + \
              df.assemble(df.dot(u_diri-u_trad,u_diri-u_trad)*meshFenics.dx(1)) ))




# VE = df.FiniteElement("Lagrange", meshFenics.ufl_cell(), 1)        
# W = df.FunctionSpace(meshFenics, VE, constrained_domain=df.SubDomain())

                  # iofe.postProcessing_simple(u, "solution.xdmf" )
# iofe.postProcessing_complete(utot, "solution_tot.xdmf", ['u','lame','vonMises'], param)
# iofe.postProcessing_complete(u, "solution_2.xdmf", ['u','lame','vonMises'], param , rename = False)
# iofe.postProcessing_complete(u, "solution_toCompare.xdmf", ['u','lame','vonMises'], param)

# Verifications
# n = df.FacetNormal(meshFenics)
 
# intU  = feut.Integral(u,meshFenics.dx,shape = (2,))
# int_un = feut.Integral(df.outer(u,n),meshFenics.ds, shape = (2,2))

# pmean = feut.Integral(p, meshFenics.dx, shape = (2,))/vol

# print(intU, int_un)
# print(pmean, Pmean)
# print(sigma_hom)
