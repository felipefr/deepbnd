from dolfin import *

class innerSquare(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 0.24 and  x[0] < 0.76 

mesh   = UnitSquareMesh(4, 4)
innerSquare = innerSquare()

# Subdomain marker
mf = MeshFunction("size_t", mesh, 2)
mf.set_all(0)
innerSquare.mark(mf, 1)

# Extract boundary mesh
bmesh = SubMesh(mesh,mf,1)
bbmesh = BoundaryMesh(bmesh, "exterior")

for e in cells(bbmesh):
    print(e.get_vertex_coordinates())

# bbmesh.entity_map(0).array()
# tt = bbmesh.data().array('parent_vertex_indices', 0)

# for x in bbmesh.coordinates():
#     x[0] = 0.0
#     x[1] = 0.0

# plot(bbmesh)

# print
# t = 0.0
# for i in range(10):
 

# ALE.move(mesh,bbmesh)
j = Mesh()
e = cpp.mesh.MeshEditor()
e.open(j,'triangle', 2, 2, 1)

print(mesh.cells())

plot(bbmesh)