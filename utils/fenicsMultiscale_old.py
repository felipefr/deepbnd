import fenicsWrapperElasticity as fela
import dolfin as df
from ufl import nabla_div
from functools import reduce
from timeit import default_timer as timer
import ioFenicsWrappers as iofe
import myCoeffClass as coef
import numpy as np

# Neumann Top and Clamped bottom
def solveNeumann(param, M, traction):
        
    M.addDirichletBC('fixed','u', df.Constant((0.0,0.0)) , 1)
    M.nameNeumannBoundary('Top',[3])    
    
    materials = M.subdomains.array().astype('int32')
    materials = materials - np.min(materials)
    
    lame = coef.getMyCoeff(materials, param, op = 'python')
    
    def sigma(u):
        return lame[0]*nabla_div(u)*df.Identity(2) + 2*lame[1]*fela.epsilon(u)
    
    # Define variational problem
    u = df.TrialFunction(M.V['u'])
    v = df.TestFunction(M.V['u'])
    T = df.Constant(tuple(list(traction)))        
    a = df.inner(sigma(u),fela.epsilon(v))*df.dx
    L = df.dot(T, v)*M.dsN['Top']    

    start = timer()
    A, b = df.assemble_system(a, L, [M.bcs['fixed']])
    end = timer()
    print('time in assembling system', end - start) # Time in seconds, e.g. 5.38091952400282
    
    u = df.Function(M.V['u'])
    U = u.vector()
    
    start = timer()
    # solve(A, U, b, 'cg', 'hypre_euclid') # solve(A, U, b , 'cg', 'ilu')
    # solve(A, U, b) # solve(A, U, b , 'cg', 'ilu')
    df.solve(A, U, b, 'cg', 'hypre_amg')

    end = timer()
    print('time in solving system', end - start) # Time in seconds, e.g. 5.38091952400282
   
    
    return u


def solveMultiscale(param, M, eps, op, linBdr = [4]):

    materials = M.subdomains.array().astype('int32')
    materials = materials - np.min(materials)
    
    lame = coef.getMyCoeff(materials, param, op = 'python')
    
    Eps = df.Constant(((eps[0,0], eps[0,1]), (eps[1,0], eps[1,1])))
    sigmaEps = lame[0]*df.tr(Eps)*df.Identity(2) + 2*lame[1]*Eps
    
    def sigma(u):
        return lame[0]*nabla_div(u)*df.Identity(2) + 2*lame[1]*fela.epsilon(u)

    

# Multiscale Linear
def solveMultiscaleLinear(param, M, eps, linBdr = [4]):
    
    for i in linBdr:
        M.addDirichletBC('fixed_' + str(i),'u', df.Constant((0.0,0.0)) , i)

    materials = M.subdomains.array().astype('int32')
    materials = materials - np.min(materials)
    
    lame = coef.getMyCoeff(materials, param, op = 'python')
    
    Eps = df.Constant(((eps[0,0], eps[0,1]), (eps[1,0], eps[1,1])))
    sigmaEps = lame[0]*df.tr(Eps)*df.Identity(2) + 2*lame[1]*Eps
    
    def sigma(u):
        return lame[0]*nabla_div(u)*df.Identity(2) + 2*lame[1]*fela.epsilon(u)
            
    # Define variational problem
    u = df.TrialFunction(M.V['u'])
    v = df.TestFunction(M.V['u'])
    a = df.inner(sigma(u),fela.epsilon(v))*M.dx
    L = -df.inner(sigmaEps, fela.epsilon(v))*M.dx    

    start = timer()
    A, b = df.assemble_system(a, L, M.bcs.values())
    end = timer()
    print('time in assembling system', end - start) # Time in seconds, e.g. 5.38091952400282
    
    u = df.Function(M.V['u'])
    U = u.vector()
    
    start = timer()
    df.solve(A, U, b, 'cg', 'hypre_amg')

    end = timer()
    print('time in solving system', end - start) # Time in seconds, e.g. 5.38091952400282
        
    return u

def solveMultiscaleLinear_zeroMean(param, M, eps, linBdr = [4]):

    VE = df.VectorElement("Lagrange", M.ufl_cell(), 1)
    RE = df.VectorElement("Real", M.ufl_cell(), 0)
    W = df.FunctionSpace(M, df.MixedElement([VE, RE]))   
    
    # V = W.sub(0).collapse()
    # R = W.sub(1).collapse()
    
    bcs = [df.DirichletBC(W.sub(0), df.Constant((0.0,0.0)) , M.boundaries, i) for i in linBdr]
    
    materials = M.subdomains.array().astype('int32')
    materials = materials - np.min(materials)
    
    lame = coef.getMyCoeff(materials, param, op = 'python')
    
    Eps = df.Constant(((eps[0,0], eps[0,1]), (eps[1,0], eps[1,1])))
    sigmaEps = lame[0]*df.tr(Eps)*df.Identity(2) + 2*lame[1]*Eps
    
    def sigma(u):
        return lame[0]*nabla_div(u)*df.Identity(2) + 2*lame[1]*fela.epsilon(u)
            
    # Define variational problem
    u1,u2,p1,p2 = df.TrialFunction(W)
    v1,v2,q1,q2 = df.TestFunction(W)
    
    
    p = df.as_vector((p1,p2))
    q = df.as_vector((q1,q2))
    u = df.as_vector((u1,u2))
    v = df.as_vector((v1,v2))
    
    a = df.inner(sigma(u),fela.epsilon(v))*M.dx +  df.inner(p,v)*M.dx + df.inner(q,u)*M.dx
    L = -df.inner(sigmaEps, fela.epsilon(v))*M.dx    
 
    start = timer()
    A, b = df.assemble_system(a, L, bcs)
    end = timer()
    print('time in assembling system', end - start) # Time in seconds, e.g. 5.38091952400282
    
    w = df.Function(W)
    
    start = timer()
    df.solve(A, w.vector(), b)
    end = timer()
    print('time in solving system', end - start) # Time in seconds, e.g. 5.38091952400282

    u, p = w.split()
    # w_ = project(w1,V.sub(0))

    # u_ = df.project(u,W.sub(0))
    
    return u, p

def solveMultiscaleMR(param, M, eps):

    VE = df.VectorElement("Lagrange", M.ufl_cell(), 1)
    RE1 = df.VectorElement("Real", M.ufl_cell(), 0)
    RE2 = df.TensorElement("Real", M.ufl_cell(), 0)
    W = df.FunctionSpace(M, df.MixedElement([VE, RE1, RE2]))   
    
    materials = M.subdomains.array().astype('int32')
    materials = materials - np.min(materials)
    
    lame = coef.getMyCoeff(materials, param, op = 'python')
    
    Eps = df.Constant(((eps[0,0], eps[0,1]), (eps[1,0], eps[1,1])))
    sigmaEps = lame[0]*df.tr(Eps)*df.Identity(2) + 2*lame[1]*Eps
    
    def sigma(u):
        return lame[0]*nabla_div(u)*df.Identity(2) + 2*lame[1]*fela.epsilon(u)
            
    # Define variational problem
    u1,u2,p1,p2, P11, P12, P21, P22 = df.TrialFunction(W)
    v1,v2,q1,q2, Q11, Q12, Q21, Q22 = df.TestFunction(W)
    
    
    p = df.as_vector((p1,p2))
    q = df.as_vector((q1,q2))
    P = df.as_vector([[P11,P12],[P21,P22]])
    Q = df.as_vector([[Q11,Q12],[Q21,Q22]])
    u = df.as_vector((u1,u2))
    v = df.as_vector((v1,v2))
    
    n = df.FacetNormal(M)
    a = df.inner(sigma(u),fela.epsilon(v))*M.dx -  df.inner(p,v)*M.dx - df.inner(q,u)*M.dx  \
        - df.inner(P,df.outer(v,n))*M.ds - df.inner(Q,df.outer(u,n))*M.ds
    L = -df.inner(sigmaEps, fela.epsilon(v))*M.dx    
 
    start = timer()
    A, b = df.assemble_system(a, L)
    end = timer()
    print('time in assembling system', end - start) # Time in seconds, e.g. 5.38091952400282
    
    w = df.Function(W)
    
    start = timer()
    df.solve(A, w.vector(), b)
    end = timer()
    print('time in solving system', end - start) # Time in seconds, e.g. 5.38091952400282

    u, p, P = w.split()
    
    # w_ = project(w1,V.sub(0))

    # u_ = df.project(u,W.sub(0))
        
    return u, p, P

def solveMultiscaleMR_allSplit(param, M, eps):

    VE = df.FiniteElement("Lagrange", M.ufl_cell(), 1)
    RE = df.FiniteElement("Real", M.ufl_cell(), 0)
    W = df.FunctionSpace(M, df.MixedElement([VE, VE, RE, RE, RE, RE, RE, RE]))   
    
    materials = M.subdomains.array().astype('int32')
    materials = materials - np.min(materials)
    
    lame = coef.getMyCoeff(materials, param, op = 'python')
    
    Eps = df.Constant(((eps[0,0], eps[0,1]), (eps[1,0], eps[1,1])))
    sigmaEps = lame[0]*df.tr(Eps)*df.Identity(2) + 2*lame[1]*Eps
    
    def sigma(u):
        return lame[0]*nabla_div(u)*df.Identity(2) + 2*lame[1]*fela.epsilon(u)
            
    # Define variational problem
    u1,u2,p1,p2, P11, P12, P21, P22 = df.TrialFunction(W)
    v1,v2,q1,q2, Q11, Q12, Q21, Q22 = df.TestFunction(W)
    
    
    p = df.as_vector((p1,p2))
    q = df.as_vector((q1,q2))
    P = df.as_vector([[P11,P12],[P21,P22]])
    Q = df.as_vector([[Q11,Q12],[Q21,Q22]])
    u = df.as_vector((u1,u2))
    v = df.as_vector((v1,v2))
    
    n = df.FacetNormal(M)
    a = df.inner(sigma(u),fela.epsilon(v))*M.dx -  df.inner(p,v)*M.dx - df.inner(q,u)*M.dx + \
        - df.inner(P,df.outer(v,n))*M.ds - df.inner(Q,df.outer(u,n))*M.ds
    L = -df.inner(sigmaEps, fela.epsilon(v))*M.dx    
 
    start = timer()
    A, b = df.assemble_system(a, L)
    end = timer()
    print('time in assembling system', end - start) # Time in seconds, e.g. 5.38091952400282
    
    w = df.Function(W)
    
    start = timer()
    df.solve(A, w.vector(), b, 'mumps')
    end = timer()
    print('time in solving system', end - start) # Time in seconds, e.g. 5.38091952400282

    u1,u2,p1,p2, P11, P12, P21, P22 = w.split(True)
    
    u = df.as_vector((u1,u2))
    P = df.as_vector([[P11,P12],[P21,P22]])
    p = df.as_vector((p1,p2))
        
    return u, p, P



def solveMultiscale_pointDirichlet(param, M, eps, uD, linBdr = [4]):

    bcs = [df.DirichletBC(M.V['u'], uD , M.boundaries, i) for i in linBdr]
    
    materials = M.subdomains.array().astype('int32')
    materials = materials - np.min(materials)
    
    lame = coef.getMyCoeff(materials, param, op = 'python')
    
    Eps = df.Constant(((eps[0,0], eps[0,1]), (eps[1,0], eps[1,1])))
    sigmaEps = lame[0]*df.tr(Eps)*df.Identity(2) + 2*lame[1]*Eps
    
    def sigma(u):
        return lame[0]*nabla_div(u)*df.Identity(2) + 2*lame[1]*fela.epsilon(u)
            
    # Define variational problem
    u = df.TrialFunction(M.V['u'])
    v = df.TestFunction(M.V['u'])
    a = df.inner(sigma(u),fela.epsilon(v))*M.dx
    L = -df.inner(sigmaEps, fela.epsilon(v))*M.dx    

    start = timer()
    A, b = df.assemble_system(a, L, bcs)
    end = timer()
    print('time in assembling system', end - start) # Time in seconds, e.g. 5.38091952400282
    
    u = df.Function(M.V['u'])
    U = u.vector()
    
    start = timer()
    df.solve(A, U, b, 'mumps')

    end = timer()
    print('time in solving system', end - start) # Time in seconds, e.g. 5.38091952400282
    

    return u

def getTotalDisplacement(u, eps, V):
    
    y = df.Function(V)
    yy = df.Expression(("A*x[0] + B*x[1]","C*x[0] + D*x[1]"), A = eps[0,0], B = eps[0,1], C = eps[1,0], D = eps[1,1], degree = 1)
    y.interpolate(yy)
   
    utot = df.Function(V)
    utot.assign(y + u) 
    
    return utot
    
def homogenisation_noMesh(u, sigma, domains, sigmaEps = df.Constant(((0.0,0.0),(0.0,0.0)))):
    M = u.function_space().mesh()
    return homogenisation(u, M, sigma, domains, sigmaEps)

def homogenisation_allSplit(u, sigma, domains, sigmaEps = df.Constant(((0.0,0.0),(0.0,0.0)))):
    M = u[0].function_space().mesh()
    return homogenisation(u, M, sigma, domains, sigmaEps)

def homogenisation(u, M, sigma, domains, sigmaEps):
    s = sigma(u)
    sigma_hom = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            sij = s[i,j] + sigmaEps[i,j]
            sigma_hom[i,j] = sum( [df.assemble(sij*M.dx(k)) for k in domains]) 

    vol = sum( [df.assemble(df.Constant(1.0)*M.dx(k)) for k in domains]) 
    
    sigma_hom = sigma_hom/vol

    return sigma_hom

class PointExpression(df.UserExpression):
    def __init__(self, u, gen):
        super().__init__()
        self.gen = gen
        self.u = u
        self.tol = 1e-10
        self.x0 = np.min(gen.x_eval[:,0])
        self.x1 = np.max(gen.x_eval[:,0])
        self.y0 = np.min(gen.x_eval[:,1])
        self.y1 = np.max(gen.x_eval[:,1])
        
    def eval(self, value, x):
        
        s = 0.0
        if(np.abs(x[1] - self.y0)<self.tol):
            # print(x, "in bottom")
            s = 0.25*(x[0] - self.x0)/(self.x1 - self.x0)
        elif(np.abs(x[0] - self.x1)<self.tol):
            s = 0.25 + 0.25*(x[1] - self.y0)/(self.y1 - self.y0)
        elif(np.abs(x[1] - self.y1)<self.tol):
            s = 0.5 + 0.25*(x[0] - self.x1)/(self.x0 - self.x1)
        elif(np.abs(x[0] - self.x0)<self.tol):
            s = 0.75 + 0.25*(x[1] - self.y1)/(self.y0 - self.y1)
        # else:
            # print(x, 'not on boundary')
        
        si = s*self.gen.npoints   
        i = int(si)
        omega = si - float(i) 
        
        for k in range(2):
            value[k] = (1.0-omega)*self.u[2*i + k]  + omega*self.u[2*(i+1) + k] 

    def value_shape(self):
        return (2,)

def getSigmaEps(param,M,eps):
    materials = M.subdomains.array().astype('int32')
    materials = materials - np.min(materials)
    
    lame = coef.getMyCoeff(materials, param, op = 'python')
    
    Eps = df.Constant(((eps[0,0], eps[0,1]), (eps[1,0], eps[1,1])))
    sigmaEps = lame[0]*df.tr(Eps)*df.Identity(2) + 2*lame[1]*Eps
    
    return sigmaEps

def getSigma(param,M):
    materials = M.subdomains.array().astype('int32')
    materials = materials - np.min(materials)
    
    lame = coef.getMyCoeff(materials, param, op = 'python')
    
    def sigma(u):
        return lame[0]*nabla_div(u)*df.Identity(2) + 2*lame[1]*fela.epsilon(u)
    
    return sigma