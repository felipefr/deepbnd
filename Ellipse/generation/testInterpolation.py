import numpy as np
import sys, os
sys.path.insert(0, '../utils/')
from timeit import default_timer as timer
import elasticity_utils as elut
import ioFenicsWrappers as iofe
import Snapshots as snap
import Generator as gene
import genericParam as gpar
import fenicsWrapperElasticity as fela
import dolfin as df

folderRB = "rbdata/"
folderSimuls = "simuls/"
Folder = 'mesh/'
MeshFile = Folder + 'mesh.xdmf'
referenceGeo = Folder + 'reference.geo'
SolutionFile = folderSimuls + "solution.hdf5"
DataFile = folderSimuls + "data.hdf5"

meshRef = fela.EnrichedMesh(MeshFile)
meshRef.createFiniteSpace('V', 'u',  'CG', 2)

mesh = fela.EnrichedMesh(MeshFile)
mesh.createFiniteSpace('V', 'u',  'CG', 2)

disp = df.Expression(('x[0]','x[1]'), degree = 2)
disp2 = df.Expression(('-x[0]','-x[1]'), degree = 2)

Idisp = df.interpolate(disp, mesh.V['u'])
Idisp2 = df.interpolate(disp2, mesh.V['u'])


pm = gpar.randomParams(seed=6, fileName = folderSimuls + 'paramFile.hdf5') # seed=6 
pm.addVariable('tx',-0.0625,0.0625)
pm.addVariable('ty',-0.0625,0.0625)
pm.addVariable('lx',-0.25,0.25)
pm.addVariable('ly',-0.25,0.25)
pm.addVariable('gamma',0.001,0.1)

pm.addSample('Train', 10, 6, 0)

defaultMeshParam = iofe.getDefaultParameters()
d = defaultMeshParam
hRef = 0.005 # 0.005
Nref = int(d['Lx1']/hRef) + 1
hRef = d['Lx1']/(Nref - 1)
d['lc'] = hRef
d['Nx1'] = Nref
d['Nx2'] = 2*Nref - 1
d['Nx3'] = Nref
d['Ny1'] = Nref
d['Ny2'] = 2*Nref - 1
d['Ny3'] = Nref

# iofe.moveMesh(mesh, meshRef, pm(0,'Train',['tx','ty','lx','ly']), defaultMeshParam)

# print(mesh.coordinates()[0::1000,:])
# print(meshRef.coordinates()[0::1000,:])


subdomain1 = df.CompiledSubDomain("near(x[0], 0.1)")
subdomain2 = df.CompiledSubDomain("near(x[0], 0.75)")
subdomain3 = df.CompiledSubDomain("near(0.25, x[1])")
subdomain4 = df.CompiledSubDomain("near(0.75, x[1])")

# boundaries = df.MeshFunction("size_t", mesh, 0)
# boundaries.set_all(0)
# subdomain1.mark(boundaries, 1)
# subdomain2.mark(boundaries, 2)
# subdomain3.mark(boundaries, 3)
# subdomain4.mark(boundaries, 4)


# bmesh = df.BoundaryMesh(mesh, "interior")


submesh = df.SubMesh(mesh, subdomain1)

print(submesh.coordinates())


df.plot(submesh, title = "Original mesh")