import dolfin as df
from ufl import nabla_div
from functools import reduce
from timeit import default_timer as timer
import core.fenics.io_wrappers as iofe
import core.fenics.my_coeff as coef
import numpy as np
from core.fenics.mesh_utils import *

def epsilon(u):
    return 0.5*(df.nabla_grad(u) + df.nabla_grad(u).T)

def sigmaLame(u, lame):
    return lame[0]*nabla_div(u)*df.Identity(2) + 2*lame[1]*epsilon(u)

def vonMises(sig):
    s = sig - (1./3)*df.tr(sig)*df.Identity(2)
    return df.sqrt((3./2)*df.inner(s, s)) 

def solveElasticityBimaterial(param, M):
    
    print(param)
    
    M.createFiniteSpace(spaceType = 'V', name = 'u', spaceFamily = 'CG', degree = 1)
    
    lame = coef.myCoeff(M.subdomains, param, degree = 1)
    
    M.addDirichletBC('clamped','u', df.Constant((0.0,0.0)) , 1)
    M.nameNeumannBoundary('Right',[2,3,4])
    M.nameRegion('all',[0,1,2,3,4,5,6,7,8,9])
    
    sigma = lambda u: sigmaLame(u,lame)
    
    # Define variational problem
    u = df.TrialFunction(M.V['u'])
    v = df.TestFunction(M.V['u'])
    T = df.Constant((-0.2,  0.0 ))
        
    a = df.inner(sigma(u),epsilon(v))*M.dxR['all']

    L = df.dot(T, v)*M.dsN['Right']
    u = df.Function(M.V['u'])

    # K = assemble(a)
    # rhs = assemble(L)
    
    # M.applyDirichletBCs(K)
    

    # solve(K,u.vector(),rhs)
    start = timer()
    A, b = df.assemble_system(a, L, [M.bcs['clamped']])
    end = timer()
    print('time in assembling system', end - start) # Time in seconds, e.g. 5.38091952400282
    
    # solver = KrylovSolver('cg', 'ilu')
    # prm = solver.parameters
    # prm.absolute_tolerance = 1E-7
    # prm.relative_tolerance = 1E-4
    # prm.maximum_iterations = 1000
    u = df.Function(M.V['u'])
    U = u.vector()
    
    start = timer()
    df.solve(A, U, b, 'cg', 'hypre_euclid') # solve(A, U, b , 'cg', 'ilu')

    end = timer()
    print('time in solving system', end - start) # Time in seconds, e.g. 5.38091952400282
   
    
    return u

def solveElasticitySimple(param, meshFile):
    
    mesh = EnrichedMesh(meshFile)
    # with XDMFFile(meshFile) as infile:
    #     infile.read(mesh)
    
    V = df.VectorFunctionSpace(mesh, 'CG' , 1)
    
    tol = 1.0e-10
    def clamped_boundary(x, on_boundary):
        return on_boundary and x[0] < tol
    
    bc = df.DirichletBC(V, df.Constant((0, 0)), clamped_boundary)
    
    lamb = param[0]
    mu = param[1]
    
    def sigma(u):
        return lamb*nabla_div(u)*df.Identity(2) + 2*mu*epsilon(u)
    
    # Define variational problem
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    f = df.Constant((0.2,  0.0 ))        
    a = df.inner(sigma(u),epsilon(v))*df.dx
    L = df.dot(f, v)*df.dx
    
    start = timer()
    A, b = df.assemble_system(a, L, [bc])
    end = timer()
    print('time in assembling system', end - start) # Time in seconds, e.g. 5.38091952400282
    
    u = df.Function(V)
    U = u.vector()
    
    
    start = timer()
    # solve(A, U, b, 'cg', 'hypre_euclid') # solve(A, U, b , 'cg', 'ilu')
    # solve(A, U, b) # solve(A, U, b , 'cg', 'ilu')
    df.solve(A, U, b, 'cg', 'hypre_amg')

    end = timer()
    print('time in solving system', end - start) # Time in seconds, e.g. 5.38091952400282
   
    
    return u


def solveElasticityBimaterial_simpler(param, M, traction):
    
    # M.createFiniteSpace(spaceType = 'V', name = 'u', spaceFamily = 'CG', degree = 2)
    
    M.addDirichletBC('clamped','u', df.Constant((0.0,0.0)) , 1)
    M.nameNeumannBoundary('Right',[2,3,4])
    
    maxBoundaries = np.max(M.boundaries.array().astype('int32')) + 1
    M.nameRegion('all',list(np.arange(maxBoundaries,maxBoundaries+10)))
    materials = M.subdomains.array().astype('int32') - maxBoundaries # to give the right range
    
    lame = coef.getMyCoeff(materials, param, op = 'cpp')
    
    def sigma(u):
        return lame[0]*nabla_div(u)*df.Identity(2) + 2*lame[1]*epsilon(u)

    # Define variational problem
    u = df.TrialFunction(M.V['u'])
    v = df.TestFunction(M.V['u'])
    T = df.Constant((traction[0], traction[1]))
    
    a = df.inner(sigma(u),epsilon(v))*M.dxR['all']
    
    L = df.dot(T, v)*M.dsN['Right']
    u = df.Function(M.V['u'])

    # K = assemble(a)
    # rhs = assemble(L)
    
    # M.applyDirichletBCs(K)
    

    # solve(K,u.vector(),rhs)
    start = timer()
    A, b = df.assemble_system(a, L, [M.bcs['clamped']])
    end = timer()
    print('time in assembling system', end - start) # Time in seconds, e.g. 5.38091952400282
    
    # solver = KrylovSolver('cg', 'ilu')
    # prm = solver.parameters
    # prm.absolute_tolerance = 1E-7
    # prm.relative_tolerance = 1E-4
    # prm.maximum_iterations = 1000
    u = df.Function(M.V['u'])
    U = u.vector()
    
    start = timer()
    df.solve(A, U, b, 'mumps') # solve(A, U, b , 'cg', 'ilu')

    end = timer()
    print('time in solving system', end - start) # Time in seconds, e.g. 5.38091952400282
   
    
    return u

def solveElasticityBimaterial_twoBCs(param, M, traction):
        
    M.addDirichletBC('fixedX','u', df.Constant(0.0) , 1 , sub = 0) # left
    M.addDirichletBC('fixedY','u', df.Constant(0.0) , 2 , sub = 1) # bottom
    
    M.nameNeumannBoundary('Right',[3,4,5])
    maxBoundaries = np.max(M.boundaries.array().astype('int32')) + 1
    # M.nameRegion('all',list(np.arange(maxBoundaries,maxBoundaries+10))) # unuseful at the moment, don't know why
    
    materials = M.subdomains.array().astype('int32') - maxBoundaries # to give the right range
    
    lame = coef.getMyCoeff(materials, param, op = 'cpp')
    
    def sigma(u):
        return lame[0]*nabla_div(u)*df.Identity(2) + 2*lame[1]*epsilon(u)

    # Define variational problem
    u = df.TrialFunction(M.V['u'])
    v = df.TestFunction(M.V['u'])
    T = df.Constant((traction[0], traction[1]))
    
    a = df.inner(sigma(u),epsilon(v))*M.dx # worked with but not with M.dx['all']
    
    L = df.dot(T, v)*M.dsN['Right']
    u = df.Function(M.V['u'])

    start = timer()
    A, b = df.assemble_system(a, L, [M.bcs['fixedX'], M.bcs['fixedY']])
    end = timer()
    print('time in assembling system', end - start) 
    
    u = df.Function(M.V['u'])
    U = u.vector()
    
    start = timer()
    # df.solve(A, U, b, 'mumps') 
    df.solve(A, U, b, 'gmres', 'amg') 

    end = timer()
    print('time in solving system', end - start) 
    
    return u


def solveElasticityBimaterial_twoBCs_2(param, M, traction):
        
    M.addDirichletBC('fixedX','u', df.Constant(0.0) , 1 , sub = 0) # left
    M.addDirichletBC('fixedY','u', df.Constant(0.0) , 2 , sub = 1) # bottom
    
    M.nameNeumannBoundary('Right',[3])
    M.nameNeumannBoundary('Top',[4])
    
    maxBoundaries = np.max(M.boundaries.array().astype('int32')) + 1
    
    materials = M.subdomains.array().astype('int32') - maxBoundaries # to give the right range
    
    lame = coef.getMyCoeff(materials, param, op = 'cpp')
    
    def sigma(u):
        return lame[0]*nabla_div(u)*df.Identity(2) + 2*lame[1]*epsilon(u)

    # Define variational problem
    u = df.TrialFunction(M.V['u'])
    v = df.TestFunction(M.V['u'])
    T1 = df.Constant((traction[0][0], traction[0][1]))
    T2 = df.Constant((traction[1][0], traction[1][1]))
    
    a = df.inner(sigma(u),epsilon(v))*M.dx # worked with but not with M.dx['all']
    
    L = df.dot(T1, v)*M.dsN['Right'] + df.dot(T2, v)*M.dsN['Top']

    u = df.Function(M.V['u'])

    start = timer()
    A, b = df.assemble_system(a, L, [M.bcs['fixedX'], M.bcs['fixedY']])
    end = timer()
    print('time in assembling system', end - start) 
    
    u = df.Function(M.V['u'])
    U = u.vector()
    
    start = timer()
    df.solve(A, U, b, 'mumps') 
    # df.solve(A, U, b, 'gmres', 'amg') 

    end = timer()
    print('time in solving system', end - start) 
    
    return u