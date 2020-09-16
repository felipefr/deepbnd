import numpy as np
import sys
sys.path.insert(0, '../utils/')
from timeit import default_timer as timer
import elasticity_utils as elut
import ioFenicsWrappers as iofe
import Snapshots as snap
import Generator as gene
import genericParam as gpar
import fenicsWrapperElasticity as fela

idSimul = 0
   
folder = "./"
radical = folder + 'prob_Ellipse10'

Label = 'Unique'

nparam = 5
ns = 1

seeds = [1,2,3]

tx = 0.0
ty = 0.0
lx = 0.0
ly = 0.0 

pm = gpar.randomParams(seed= seeds[idSimul], name = 'pm', fileName = radical + 'ParamFile.hdf5') 
pm.addVariable('tx',tx,tx)
pm.addVariable('ty',ty,ty)
pm.addVariable('lx',lx,lx)
pm.addVariable('ly',ly,ly)
pm.addVariable('gamma',0.001,0.001)

pm.addSample(Label, ns)
pm.write('w')

pc = gpar.genericParams()
pc.addVariable('nu', 0.3)
pc.addVariable('E', 1.0)

pd = gpar.derivedParams(pm,pc,name = 'pd')
pd.addVariable('lamb1', None, ['nu','E'] , elut.eng2lambPlane)
pd.addVariable('lamb2', ['gamma'],['nu','E'] , lambda x,y,z: elut.eng2lambPlane(y,x*z))
pd.addVariable('mu1', None, ['nu','E'] , elut.eng2mu)
pd.addVariable('mu2', ['gamma'],['nu','E'] , lambda x,y,z: elut.eng2mu(y,x*z))

# Creating reference Mesh
defaultMeshParam = iofe.getDefaultParameters_given_hRef(hRef = 0.01)
defaultMeshParam['theta'] = np.pi/4
defaultMeshParam['alpha'] = np.pi/4
defaultMeshParam['x0'] = 0.7
defaultMeshParam['y0'] = 0.7 
defaultMeshParam['x1'] = 0.3
defaultMeshParam['y1'] = 0.3
meshFile = folder + 'mesh_Ellipse10.xdmf'
iofe.exportMeshXDMF_fromReferenceGeo(defaultMeshParam, referenceGeo = 'mesh/referenceGeo_twoEllipses.geo', meshFile = meshFile)

# traction = np.array([0.1,0.0])
# femData = {'defaultMeshParam': defaultMeshParam , 
#             'problem' : lambda x,y: fela.solveElasticityBimaterial_twoBCs(x, y, traction).vector().get_local(),
#            'fespace' : {'spaceType' : 'V', 'name' : 'u', 'spaceFamily' : 'CG', 'degree' : 2} }

traction = [np.array([0.1,0.0]) , np.array([0.0,0.0])]
femData = {'defaultMeshParam': defaultMeshParam , 
            'problem' : lambda x,y: fela.solveElasticityBimaterial_twoBCs_2(x, y, traction).vector().get_local(),
           'fespace' : {'spaceType' : 'V', 'name' : 'u', 'spaceFamily' : 'CG', 'degree' : 2} }


s = snap.Snapshots(radical, femData, mode = 'ignore')
s.meshFile = meshFile
s.getMapParamToDomain = snap.getMapParamToDomain2

s.registerParameters([pm,pd])

s.buildSnapshots(Label)

s.exportVTK('paraview_Ellipse10.xdmf', Label , indexes = np.arange(ns)  )

# s.posProcessingStress(Label)
