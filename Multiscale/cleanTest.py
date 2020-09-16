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

# mesh =  gmsh.ellipseMesh(ellipseData = [(0.5,0.2,0.2,0.5,0.0),(0.5,0.8,0.2,0.3,0.0)], ifPeriodic = False) () 
# mesh.write('test.vtk')

# meshGMSH =  gmsh.ellipseMesh(ellipseData = geni.circularRegular(0.07, 0.1, 4, 4), lcar = 0.02, ifPeriodic = False) 
# mesh = meshGMSH(['-algo','del2d'])
# mesh.write('test.vtk')

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
# ellipseData = geni.circularRegular(0.07, 0.1, 4, 4)
# meshGMSH =  gmsh.ellipseMesh( ellipseData , Lx, Ly, lcar, ifPeriodic) 

ellipseData = geni.circularRegular2Regions(r0, r1, NxL, NyL, Lx, Ly, offset, ordered = True)
meshGMSH =  gmsh.ellipseMesh2DomainsPhysicalMeaning(x0L, y0L, LxL, LyL, NxL*NyL, ellipseData , Lx, Ly, lcar, ifPeriodic) 

# meshGMSH.generate(['-algo','del2d']) # this raises a error for lower version of meshio

meshGMSH.write('test3.geo','geo')
os.system('gmsh -2 -algo del2d -format msh2 test3.geo')

# meshGMSH.write('test5.xdmf', 'fenics')

# iofe.exportMeshXDMF_fromGMSH('test3.msh', 'test7.xdmf', labels = {'line' : 'faces', 'triangle' : 'regions'})
    
# iofe.exportMeshHDF5_fromGMSH('test3.msh', 'test6.xdmf', labels = {'line' : 'faces', 'triangle' : 'regions'})

os.system("dolfin-convert test3.msh test7.xml")

traction = np.array([0.0,10.0])
femData = { 'meshFile' : 'test7.xml',
            # 'problem' : lambda a,b,c,d,e,f : fmts.solveMultiscale(a, b, c, d, e, f),
            'problem' : lambda a,b,c,d,e : fmts.solveMultiscaleMR_3(a, b, c, d, e),
            'fespace' : {'spaceType' : 'V', 'name' : 'u', 'spaceFamily' : 'CG', 'degree' : 1} }



meshFenics = fela.EnrichedMesh(femData['meshFile'])
meshFenics.createFiniteSpace(**femData['fespace'])
        
param = np.array([ [10.0,15.0], [20.0,30.0], [10.0,15.0], [20.0,30.0] ])
# u = fmts.solveNeumann(param,meshFenics,traction)

eps = np.zeros((2,2))
eps[0,0] = 0.1

u, p, P, sigma_hom = femData['problem'](param, meshFenics, eps, [0,1], [0,1,2,3], )
# u, sigma_hom = femData['problem'](param, meshFenics, eps, [0,1], [0,1,2,3], )
# utot, u, sigma_hom, sigma_homL = femData['problem'](param, meshFenics, eps, [0,1], [0,1,2,3], 'linear')

# iofe.postProcessing_complete(u, "solution_MR.xdmf", ['u'], param , rename = False)


n = df.FacetNormal(meshFenics)
un = df.outer(u,n)

intU1  = df.assemble(u[0]*meshFenics.dx)
intU2  = df.assemble(u[1]*meshFenics.dx)

int_un11 = df.assemble(un[0,0]*meshFenics.ds)
int_un12 = df.assemble(un[0,1]*meshFenics.ds)
int_un21 = df.assemble(un[1,0]*meshFenics.ds)
int_un22 = df.assemble(un[1,1]*meshFenics.ds)

P11 = df.assemble(P[0,0]*meshFenics.dx)/df.assemble(df.Constant(1.0)*meshFenics.dx)
P12 = df.assemble(P[0,1]*meshFenics.dx)/df.assemble(df.Constant(1.0)*meshFenics.dx)
P21 = df.assemble(P[1,0]*meshFenics.dx)/df.assemble(df.Constant(1.0)*meshFenics.dx)
P22 = df.assemble(P[1,1]*meshFenics.dx)/df.assemble(df.Constant(1.0)*meshFenics.dx)

p1 = df.assemble(p[0]*meshFenics.dx)/df.assemble(df.Constant(1.0)*meshFenics.dx)
p2 = df.assemble(p[1]*meshFenics.dx)/df.assemble(df.Constant(1.0)*meshFenics.dx)


print(intU1, intU2, int_un11, int_un12, int_un21, int_un22)
print(p1,p2, P11, P12, P21, P22)
print(sigma_hom)

# g = gmts.displacementGeneratorBoundary(x0L,y0L,LxL,LyL,10)
g = gmts.displacementGeneratorBoundary(0.0,0.0,1.0,1.0,200)

uBound = g(meshFenics,u)

    
uDirichlet = fmts.PointExpression(uBound, g)
utot, u, sigma, sigmaL = fmts.solveMultiscale_pointDirichlet(param, meshFenics, eps, uDirichlet, [0,1], [0,1,2,3])
print(sigma,sigmaL)
print(P11, P12, P21, P22)


iofe.postProcessing_complete(u, "solution_Dirich.xdmf", ['u'], param)


# VE = df.FiniteElement("Lagrange", meshFenics.ufl_cell(), 1)        
# W = df.FunctionSpace(meshFenics, VE, constrained_domain=df.SubDomain())

                  # iofe.postProcessing_simple(u, "solution.xdmf" )
# iofe.postProcessing_complete(utot, "solution_tot.xdmf", ['u','lame','vonMises'], param)
# iofe.postProcessing_complete(u, "solution_2.xdmf", ['u','lame','vonMises'], param , rename = False)
# iofe.postProcessing_complete(u, "solution_toCompare.xdmf", ['u','lame','vonMises'], param)


# folder = "./"
# radical = folder + 'prob_Ellipse10'

# s.exportVTK('paraview_Ellipse10.xdmf', Label , indexes = np.arange(ns)  )

# s.posProcessingStress(Label)
