from dolfin import *
import numpy as np
import sympy as sy
from sympy.printing import ccode
import matplotlib.pyplot as plt

#build reference solution
x_ = sy.Symbol('x[0]')
y_ = sy.Symbol('x[1]')
uref_ = sy.sin(x_ + 2*y_)
# rhs_ = - sy.diff(uref_, x_, x_) - sy.diff(uref_, y_, y_)


uref_exp = Expression(ccode(uref_), degree=10)
# rhs = Expression(ccode(rhs_), degree=10)

uref_ = lambda x : sin(x[0] + 2 * x[1])
rhs_ = lambda u: -div(grad(u))


err_h1 = []
err_h1_exp = []
iMax = 5
nElem = [5*2**ii for ii in range(iMax)]
for i,N in enumerate(nElem):
    mesh = UnitSquareMesh(N, N)
    x = SpatialCoordinate(mesh)
    
    uref = uref_(x)
    rhs = rhs_(uref)
    
    U = FunctionSpace(mesh, "CG", 1)

    quad_element1 = FiniteElement(family = "Quadrature",
                                  cell = mesh.ufl_cell(),
                                  degree = 1,
                                  quad_scheme="canonical")

    quad_element2 = FiniteElement(family = "Quadrature",
                                  cell = mesh.ufl_cell(),
                                  degree = 1,
                                  quad_scheme="vertex")
    
    Q1 = FunctionSpace(mesh, quad_element1)
    Q2 = FunctionSpace(mesh, quad_element2)


    u = TrialFunction(U)
    v = TestFunction(U)

    a = inner(grad(u), grad(v))*dx
    L = rhs*v*dx
    bc = DirichletBC(U, uref, "on_boundary")
    
    uu = Function(U)

    solve(a == L, uu, bc)
    q1 = interpolate(uu, Q1)
    q2 = interpolate(uu, Q2)
    
    # err_h1.append(errornorm(interpolate(uref,U), uu, 'L2'))
    error = lambda du : np.sqrt(assemble(du*du*dx, form_compiler_parameters = {'quadrature_degree': 10}))
    
    err_h1.append(error(uref - uu))
    err_h1_exp.append(errornorm(uref_exp, uu, 'L2'))

    plt.figure(i)
    plt.subplot('121')
    plot(uu)
    plot(mesh,color='k',linewidth = 0.1)
    plot(refine(mesh),color='k',linewidth = 0.1)
    
    # plt.subplot('122')
    # plot(q2)
    # plot(mesh,color='k',linewidth = 0.1)
    
    # print(np.linalg.norm(q1.vector()[:] - q2.vector()[:]))
    
print("H1 error: ", err_h1)

#estimate rate of convergence
errh1 = np.array(err_h1) 
errh1_exp = np.array(err_h1_exp) 
print((errh1 - errh1_exp)/errh1_exp)
quotient = errh1[:-1]/errh1[1:]
rateh1 = np.log(quotient)/np.log(2)
print("rate: ", rateh1)

quotient = errh1_exp[:-1]/errh1_exp[1:]
rateh1 = np.log(quotient)/np.log(2)
print("rate: ", rateh1)


# from dolfin import *
# mesh = UnitSquareMesh(4,4)


# finemesh = adapt(mesh)
# DGfine = VectorFunctionSpace(finemesh, "DG", 0)
# field = Function(DGfine)


# parameters["form_compiler"]["cpp_optimize"] = True
# ffc_options = {"optimize": True, \
#            "eliminate_zeros": False, \
#            "precompute_basis_const": True, \
#            "precompute_ip_const": True}
