from __future__ import print_function
import numpy as np
from fenics import *
from dolfin import *
from ufl import nabla_div
import matplotlib.pyplot as plt

import elasticity_utils as elut


factorForce = 0.1
bodyForce = lambda theta: (np.cos(theta),factorForce*np.sin(theta)) 


def solveElasticityBimaterial(param):

    h = 1.0; w = 1.0
    # domain = Rectangle(Point(-w, -h), Point(w, h))
    
    mesh = RectangleMesh(Point(-w, -h), Point(w, h), 200, 200, 'right/left')
    # mesh = generate_mesh(domain,30)
    V = VectorFunctionSpace(mesh, 'CG', 2)

    
    tol = 1E-14
    lamb1 , mu1 = param[0]
    lamb2 , mu2 = param[1]
    
    r2 = 0.25
    
    lamb = Expression('x[1]*x[1] + x[0]*x[0] <= r2 + tol ? lamb1 : lamb2', degree=0,
                   tol=tol, lamb1=lamb1, lamb2=lamb2, r2 = r2)
    
    mu = Expression('x[1]*x[1] + x[0]*x[0] > r2 ? mu1 : mu2', degree=0,
                   tol=tol, mu1=mu1, mu2=mu2, r2=r2)
    
    
    def clamped_boundary(x, on_boundary):
        return on_boundary and x[0] < -w + tol

    bc = DirichletBC(V, Constant((0.,0.0)), clamped_boundary)  
    
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
    T = Constant((-0.5,  0.0 ))
    a = inner(sigma(u), epsilon(v))*dx
    L = dot(f, v)*dx + dot(T, v)*ds(1)
    
    # angleGravity = 0.0
    # f.assign( Constant( bodyForce(angleGravity) )) 

    K = assemble(a)
    rhs = assemble(L)
    
    bc.apply(K,rhs)
    
    u = Function(V)
    solve(K,u.vector(),rhs)
    
    
    s = sigma(u) - (1./3)*tr(sigma(u))*Identity(d)
    von_Mises = sqrt(((3./2)*inner(s, s))) 
    
    W =  FunctionSpace(mesh, 'DG',0)
    von_Mises = project(von_Mises, W)
    
    sigma_xx = project(sigma(u)[0,0], W)
    sigma_xy = project(sigma(u)[0,1], W)
    sigma_yy = project(sigma(u)[1,1], W)
    
    fileResults = XDMFFile("output.xdmf")
    fileResults.parameters["flush_output"] = True
    fileResults.parameters["functions_share_mesh"] = True

    # nu1, E1 = elut.composition(elut.lame2youngPoisson, elut.lameStar2lame)(lamb1, mu1)    
    # nu2, E2 = elut.composition(elut.lame2youngPoisson, elut.lameStar2lame)(lamb2, mu2) 
    
    # young = Expression('x[1]*x[1] + x[0]*x[0] <= r2 + tol ? E1 : E2', degree=0,
    #            tol=tol, E1=E1, E2=E2, r2 = r2)



    u.rename('u', 'displacements at nodes')
    von_Mises.rename('von_Mises', 'Von Mises stress')
    sigma_xx.rename('sigma_xx', 'Cauchy stress in component xx')
    sigma_xy.rename('sigma_xy', 'Cauchy stress in component xy')
    sigma_yy.rename('sigma_yy', 'Cauchy stress in component yy')
    
    fileResults.write(u,0.)
    fileResults.write(von_Mises,0.)    
    fileResults.write(sigma_xx,0.)
    fileResults.write(sigma_xy,0.)
    fileResults.write(sigma_yy,0.)


    
    # f = File('ex_bimaterial.xdmf')
    # f << u
    # f << von_Mises
    
    return u


nu1 = 0.3
nu2 = 0.3

E1 = 1.0
E2 = 100.0

lamb1, mu1 = elut.youngPoisson2lame_planeStress(nu1, E1)
lamb2, mu2 = elut.youngPoisson2lame_planeStress(nu2, E2)

u= solveElasticityBimaterial([(lamb1, mu1), (lamb2, mu2)])



