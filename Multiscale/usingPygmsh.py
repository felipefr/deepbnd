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

# meshGMSH.write('test3.geo','geo')
# os.system('gmsh -2 -algo del2d test3.geo')

# meshGMSH.write('test5.xdmf', 'fenics')

# iofe.exportMeshHDF5_fromGMSH('test3.msh', 'test3.xdmf', labels = {'line' : 'faces', 'triangle' : 'regions'})

# meshVTK = pv.read('test.vtk')
# meshVTK.plot(show_edges=True)

# p = pv.Plotter()
# p.add_mesh(meshVTK, color = True, show_edges = True)
# p.show(cpos=[0, 0, 0.5])

# idSimul = 0
   
# folder = "./"
# radical = folder + 'prob_Ellipse10'

# Label = 'Unique'

# nparam = 5
# ns = 1

# seeds = [1,2,3]

# tx = 0.0
# ty = 0.0
# lx = 0.0
# ly = 0.0 

# pm = gpar.randomParams(seed= seeds[idSimul], name = 'pm', fileName = radical + 'ParamFile.hdf5') 
# pm.addVariable('tx',tx,tx)
# pm.addVariable('ty',ty,ty)
# pm.addVariable('lx',lx,lx)
# pm.addVariable('ly',ly,ly)
# pm.addVariable('gamma',0.001,0.001)

# pm.addSample(Label, ns)
# pm.write('w')

# pc = gpar.genericParams()
# pc.addVariable('nu', 0.3)
# pc.addVariable('E', 1.0)

# pd = gpar.derivedParams(pm,pc,name = 'pd')
# pd.addVariable('lamb1', None, ['nu','E'] , elut.eng2lambPlane)
# pd.addVariable('lamb2', ['gamma'],['nu','E'] , lambda x,y,z: elut.eng2lambPlane(y,x*z))
# pd.addVariable('mu1', None, ['nu','E'] , elut.eng2mu)
# pd.addVariable('mu2', ['gamma'],['nu','E'] , lambda x,y,z: elut.eng2mu(y,x*z))

# Creating reference Mesh
# defaultMeshParam = iofe.getDefaultParameters_given_hRef(hRef = 0.01)
# defaultMeshParam['theta'] = np.pi/4
# defaultMeshParam['alpha'] = np.pi/4
# defaultMeshParam['x0'] = 0.7
# defaultMeshParam['y0'] = 0.7 
# defaultMeshParam['x1'] = 0.3
# defaultMeshParam['y1'] = 0.3
# meshFile = folder + 'mesh_Ellipse10.xdmf'
# iofe.exportMeshXDMF_fromReferenceGeo(defaultMeshParam, referenceGeo = 'mesh/referenceGeo_twoEllipses.geo', meshFile = meshFile)


# traction = [np.array([0.1,0.0]) , np.array([0.0,0.0])]
# femData = {'defaultMeshParam': defaultMeshParam , 
#             'problem' : lambda x,y: fela.solveElasticityBimaterial_twoBCs_2(x, y, traction).vector().get_local(),
#            'fespace' : {'spaceType' : 'V', 'name' : 'u', 'spaceFamily' : 'CG', 'degree' : 2} }


# s = snap.Snapshots(radical, femData, mode = 'ignore')
# s.meshFile = meshFile
# s.getMapParamToDomain = snap.getMapParamToDomain2

# s.registerParameters([pm,pd])

# s.buildSnapshots(Label)

# s.exportVTK('paraview_Ellipse10.xdmf', Label , indexes = np.arange(ns)  )

# s.posProcessingStress(Label)
