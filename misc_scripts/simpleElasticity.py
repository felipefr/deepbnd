from __future__ import print_function
import numpy as np
from fenics import *
from dolfin import *
from ufl import nabla_div
import matplotlib.pyplot as plt

import elasticity_utils as elut


factorForce = 0.1
bodyForce = lambda theta: (np.cos(theta),factorForce*np.sin(theta)) 


def solveElasticity(param):

    h = 1.0; w = 1.0
    # domain = Rectangle(Point(-w, -h), Point(w, h))
    
    mesh = RectangleMesh(Point(0.0, 0.0), Point(w, h), 4, 4, 'right/left')
    V = VectorFunctionSpace(mesh, 'CG', 1)

    
    tol = 1E-14
    lamb , mu = param
    
    def clamped_boundary(x, on_boundary):
        return on_boundary and x[0] < tol

    bc = DirichletBC(V, Constant((0.,0.)), clamped_boundary)  
    
    class rightBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] > w - tol
        
    bed= rightBoundary()
    
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
    boundaries.set_all(0)
    bed.mark(boundaries, 1)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    
    def epsilon(u):
        return 0.5*(nabla_grad(u) + nabla_grad(u).T)
    
    def sigma(u):
        return lamb*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)
    
    # Define variational problem
    u = TrialFunction(V)
    d = u.geometric_dimension()  # space dimension
    v = TestFunction(V)
    f = Constant((0, 0))
    T = Constant((0.5,  0.0 ))
    a = inner(sigma(u), epsilon(v))*dx
    L = dot(f, v)*dx + dot(T, v)*ds(1)

    K = assemble(a)
    rhs = assemble(L)
    
    bc.apply(K,rhs)
    
    u = Function(V)
    solve(K,u.vector(),rhs)

    fileResults = XDMFFile("output.xdmf")
    fileResults.parameters["flush_output"] = True
    fileResults.parameters["functions_share_mesh"] = True

    u.rename('u', 'displacements at nodes')
    
    fileResults.write(u,0.)

    # f = File('ex_bimaterial.xdmf')
    # f << u
    # f << von_Mises
    
    return u, V


nu = 0.3
E = 1.0


lamb, mu = elut.youngPoisson2lame_planeStress(nu, E)

u , V= solveElasticity([lamb, mu])

XY = V.tabulate_dof_coordinates()

v = u.compute_vertex_values(V.mesh()).reshape((2,25))
v2d = vertex_to_dof_map(V) # used to take a vector in its dofs and transform to dofs according to node numbering in mesh
d2v = dof_to_vertex_map(V) # used to transform node coordinates of mesh into computational nodes coordinates

print(v2d.shape)
print(d2v.shape)

# for i, xy in enumerate( XY[0::2,:]):
#     print(u(xy), u.vector().get_local()[2*i:2*(i+1)])

# print('/n /n')
# for i, xy in enumerate(V.mesh().coordinates()):
#     print(u(xy), v[:,i])
    
for i, xy in enumerate( XY[0::2,:]):
    print(u.vector().get_local()[v2d[2*i:2*(i+1)]] , v[:,i])

nodes = V.mesh().coordinates().flatten()

print(XY[::2,:])

print(V.mesh().coordinates()[(d2v[::2]/2.0).astype('int'),:])  # computational node 

