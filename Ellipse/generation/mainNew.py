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

idSimul = 5 
   
folder = "simuls2/"
radical = folder + 'prob1_' + str(idSimul)

Label = 'Unique'

nparam = 5
ns = 2000

seeds = [8,9,10,11,12,14]

pm = gpar.randomParams(seed= seeds[idSimul], name = 'pm', fileName = radical + 'ParamFile.hdf5') 
pm.addVariable('tx',-0.0625,0.0625)
pm.addVariable('ty',-0.0625,0.0625)
pm.addVariable('lx',-0.25,0.25)
pm.addVariable('ly',-0.25,0.25)
pm.addVariable('gamma',0.001,0.1)

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
defaultMeshParam = iofe.getDefaultParameters_given_hRef(hRef = 0.005)

traction = np.array([0.0,0.05])
femData = {'defaultMeshParam': defaultMeshParam , 
           'problem' : lambda x,y: fela.solveElasticityBimaterial_simpler(x,y,traction).vector().get_local(),
           'fespace' : {'spaceType' : 'V', 'name' : 'u', 'spaceFamily' : 'CG', 'degree' : 2} }

s = snap.Snapshots(radical, femData, mode = 'ignore')
s.meshFile = folder + 'mesh.xdmf'

s.registerParameters([pm,pd])

start = timer()

# s.buildSnapshots(Label)

# s.posProcessingStress(Label)

g1 = gene.stressGenerator('Y',Label, 10, 10,  0, ns)
g2 = gene.displacementGenerator('X', Label, ['Right','Bottom','Top'],[10,10,10], 0.05, 0, ns)
s.generateData([g2,g1],'w')

end = timer()

print(end - start) # Time in seconds, e.g. 5.38091952400282