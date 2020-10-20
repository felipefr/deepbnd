import fenicsWrapperElasticity as fela
import dolfin as df
from ufl import nabla_div
from functools import reduce
from timeit import default_timer as timer
import ioFenicsWrappers as iofe
import myCoeffClass as coef
import numpy as np
import fenicsUtils as feut
import generatorMultiscale as gmts
import fenicsUtils as feut
from PointExpression import *

def getBMrestriction(g,M):
    m = 2*g.npoints
    mm = m/8  
    
    BM = np.zeros((m,4))
    normal1 = np.zeros(2)
    normal2 = np.zeros(2)
    ej = np.zeros(2)
    
    h = g.x_eval[1,0] - g.x_eval[0,0]
    print("height", h)
    
    for i in range(m):
        I = int(i/2)
        j = i%2 
        ej[j] = 1.0
            
        if(I==0):
            normal1[0] = -1.0
            normal2[1] = -1.0
        elif(I<mm):
            normal1[1] = -1.0
            normal2[1] = -1.0
        elif(I==mm):
            normal1[1] = -1.0
            normal2[0] = 1.0
        elif(I<2*mm):
            normal1[0] = 1.0
            normal2[0] = 1.0            
        elif(I==2*mm):
            normal1[0] = 1.0
            normal2[1] = 1.0            
        elif(I<3*mm):
            normal1[1] = 1.0
            normal2[1] = 1.0            
        elif(I==3*mm):
            normal1[1] = 1.0
            normal2[0] = -1.0                
        elif(I<4*mm):
            normal1[0] = -1.0
            normal2[0] = -1.0
            
        BMI = 0.5*h*( np.outer(ej,normal1 + normal2) )
        BM[i,:] = BMI.reshape(4)
        
        ej.fill(0.0)
        normal1.fill(0.0)
        normal2.fill(0.0)
        
    return BM
                    
def pod_customised(S, op = '', others = []):
    
    Mhsqrt = []
    Mh = []
    
    if(op == 'L2'):
        g = others[0]
        # mesh = others[1]
        # Mh = getMassMatrixBoundary(g,mesh)
        Mh = getMassMatrixBoundary_fast(g)
        U,sigma,UT = np.linalg.svd(Mh)
        Mhsqrt = U@np.diag(np.sqrt(sigma))@UT
        S = Mhsqrt@S
        
    Wbasis, sig, VT = np.linalg.svd(S)
    
    if(op == 'L2'):
        MhsqrtInv = U@np.diag(sigma**-0.5)@UT
        Wbasis = MhsqrtInv@Wbasis
    
    
    return Wbasis, sig, Mhsqrt, Mh

def pod_customised_2(S, op = '', others = []):
    
    Mhsqrt = []
    Mh = []
    
    if(op == 'L2'):
        g = others[0]
        # mesh = others[1]
        # Mh = getMassMatrixBoundary(g,mesh)
        Mh = getMassMatrixBoundary_fast(g)
        U,sigma,UT = np.linalg.svd(Mh)
        Mhsqrt = U@np.diag(sigma**0.5)@UT
        S = Mhsqrt@S
        
    Wbasis, sig, VT = np.linalg.svd(S)
    
    if(op == 'L2'):
        MhsqrtInv = U@np.diag(sigma**-0.5)@UT
        WbasisBar = MhsqrtInv@Wbasis
    
    
    return WbasisBar, Wbasis, sig, Mhsqrt, Mh, MhsqrtInv

def getMassMatrixBoundary_fast(g):
    m = 2*g.npoints

    h = g.x_eval[1,0] - g.x_eval[0,0]
    
    Mh = (2*h/3)*np.eye(m) + (h/6)*np.eye(m,k=2) + (h/6)*np.eye(m,k=-2)
    Mh[0,-2] = h/6
    Mh[1,-1] = h/6
    Mh[-2,0] = h/6
    Mh[-1,1] = h/6

    print("Mass matrix boundary computed", Mh.max)
    return Mh
        

def getMassMatrixBoundary(g, M):
    m = 2*g.npoints
    print('points', m)
    
    Mh = np.zeros((m,m))
    
    ei = np.zeros(m)
    ej = np.zeros(m)
    
    for i in range(m):
        ei[i] = 1.0
        fei = PointExpression(ei, g)
        
        # for j in [(i-1)%m,i,(i+1)%m] :
        for j in range(i,m):
            ej[j] = 1.0
            fej = PointExpression(ej, g)
            Mh[i,j] = df.assemble(df.inner(fei,fej)*M.ds)
            ej[j] = 0.0
            # Mh[j,i] = Mh[i,j]
    
        ei[i] = 0.0
    
    Mh = 0.5*(Mh + Mh.T)
    print("Mass matrix boundary computed", Mh.max)
    return Mh
        
        
def getSigma_SigmaEps(param,M,eps,op):
    materials = M.subdomains.array().astype('int32')
    materials = materials - np.min(materials)
    print(materials.max(), materials.min())
    
    lame = coef.getMyCoeff(materials, param, op = op)
    
    Eps = df.Constant(((eps[0,0], 0.5*(eps[0,1] + eps[1,0])), (0.5*(eps[0,1] + eps[1,0]), eps[1,1])))
    sigmaEps = lame[0]*df.tr(Eps)*df.Identity(2) + 2*lame[1]*Eps
    
    def sigma(u):
        return lame[0]*nabla_div(u)*df.Identity(2) + 2*lame[1]*fela.epsilon(u)

    return sigma, sigmaEps

def homogenisedTangent(param, M, op, linBdr = [4]):

    Chom = np.zeros((3, 3))
    eps = np.zeros((2,2))
    
    vol = df.assemble(df.Constant(1.0)*M.dx)
    
    for i in range(3):
        j, k = [[0,0], [0,1] ,[1,1]][i]
        eps[j,k] = 1.0
        eps = 0.5*(eps + eps.T)
        
        sol = solveMultiscale(param, M, eps, op, linBdr )
        
        if(op == 'linear'):
            u = sol
        else:
            u = sol.split()[0]
            
        sigma, sigmaEps = getSigma_SigmaEps(param, M, eps)

        Sigma_Int = feut.Integral(sigmaEps + sigma(u),M.dx,shape=(2,2))/vol
        
        for l in range(3):
            j, k = [[0,0], [0,1] ,[1,1]][l]
            Chom[l, :] = Sigma_Int[j,k]
    
    
    lmbda_hom = Chom[0, 1]
    mu_hom = Chom[2, 2]
    
    E_hom = mu_hom*(3*lmbda_hom + 2*mu_hom)/(lmbda_hom + mu_hom)
    nu_hom = lmbda_hom/(lmbda_hom + mu_hom)/2
    
    return Chom, E_hom, nu_hom

def solveMultiscale(param, M, eps, op, others = [[4]]):

    sigma, sigmaEps = getSigma_SigmaEps(param,M,eps)    
    
    if(op == 'linear'):
        linBdr = others[0]
        a,f,bcs,V = formulationMultiscaleBCdirich(M, sigma, sigmaEps, linBdr, uD = df.Constant((0.0,0.0)))
    elif(op == 'linear_zero'):
        linBdr = others[0]
        a,f,bcs,V = formulationMultiscaleBCdirich_zeroMean(M, sigma, sigmaEps, linBdr, uD = df.Constant((0.0,0.0)))
    elif(op == 'MR'):
        a,f,bcs,V = formulationMultiscaleMR(M, sigma, sigmaEps)
    elif(op == 'BCdirich'): 
        linBdr = others[0]
        uD_ = others[1]
        g = gmts.displacementGeneratorBoundary(0.0,0.0,1.0,1.0, int((uD_.shape[0] + 8)/8) )
        uD = PointExpression(uD_, g)

        a,f,bcs,V = formulationMultiscaleBCdirich_zeroMean(M, sigma, sigmaEps, linBdr, uD)
    
    elif(op == 'periodic'):
        a,f,bcs,V = formulationMultiscale_periodic(M, sigma, sigmaEps)
        
    elif(op == 'POD'):    
        a,f,bcs,V = formulationMultiscalePOD(M, sigma, sigmaEps, Ubasis = others[0], x_eval = others[1], alpha = others[2])
    
            
    start = timer()
    
    A, b = df.assemble_system(a, f, bcs)    
    sol = df.Function(V)
    df.solve(A, sol.vector(), b)

    end = timer()
    print('time in solving system', end - start) # Time in seconds, e.g. 5.38091952400282

    return sol

# Multiscale Linear
def formulationMultiscaleBCdirich(M, sigma, sigmaEps, linBdr, uD):
    
    for i in linBdr:
        M.addDirichletBC('fixed_' + str(i),'u', uD , i)
            
    u = df.TrialFunction(M.V['u'])
    v = df.TestFunction(M.V['u'])
    a = df.inner(sigma(u),fela.epsilon(v))*M.dx
    L = -df.inner(sigmaEps, fela.epsilon(v))*M.dx    

    return a, L, M.bcs.values() , M.V['u']

def formulationMultiscaleBCdirich_zeroMean(M, sigma, sigmaEps, linBdr, uD):

    VE = df.VectorElement("Lagrange", M.ufl_cell(), 1)
    RE = df.VectorElement("Real", M.ufl_cell(), 0)
    W = df.FunctionSpace(M, df.MixedElement([VE, RE]))   
        
    bcs = [df.DirichletBC(W.sub(0), uD , M.boundaries, i) for i in linBdr]
                
    # Define variational problem
    u1,u2,p1,p2 = df.TrialFunction(W)
    v1,v2,q1,q2 = df.TestFunction(W)
        
    p = df.as_vector((p1,p2))
    q = df.as_vector((q1,q2))
    u = df.as_vector((u1,u2))
    v = df.as_vector((v1,v2))
    
    a = df.inner(sigma(u),fela.epsilon(v))*M.dx +  df.inner(p,v)*M.dx + df.inner(q,u)*M.dx
    L = -df.inner(sigmaEps, fela.epsilon(v))*M.dx    
 
    return a, L, bcs, W


def formulationMultiscaleMR(M, sigma, sigmaEps):

    VE = df.VectorElement("Lagrange", M.ufl_cell(), 1)
    RE1 = df.VectorElement("Real", M.ufl_cell(), 0)
    RE2 = df.TensorElement("Real", M.ufl_cell(), 0)
    W = df.FunctionSpace(M, df.MixedElement([VE, RE1, RE2]))   
    
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
 
    return a, L, [], W 

def formulationMultiscale_periodic(M, sigma, sigmaEps):
    VE = df.VectorElement("Lagrange", M.ufl_cell(), 1)
    RE1 = df.VectorElement("Real", M.ufl_cell(), 0)
    W = df.FunctionSpace(M, df.MixedElement([VE, RE1]), constrained_domain = PeriodicBoundary() )   
    
    # Define variational problem
    u1,u2,p1,p2 = df.TrialFunction(W)
    v1,v2,q1,q2 = df.TestFunction(W)
    
    p = df.as_vector((p1,p2))
    q = df.as_vector((q1,q2))
    u = df.as_vector((u1,u2))
    v = df.as_vector((v1,v2))
    
    n = df.FacetNormal(M)
    a = df.inner(sigma(u),fela.epsilon(v))*M.dx -  df.inner(p,v)*M.dx - df.inner(q,u)*M.dx
    L = -df.inner(sigmaEps, fela.epsilon(v))*M.dx    
 
    return a, L, [], W 

def formulationMultiscalePOD(M, sigma, sigmaEps, Ubasis, x_eval, alpha):

    # N = Ubasis.shape[1]    
    # m = Ubasis.shape[0]
    
    VE = df.VectorElement("Lagrange", M.ufl_cell(), 1)
    RE1 = df.VectorElement("Real", M.ufl_cell(), 0)
    RE2 = df.VectorElement("Real", M.ufl_cell(), 0)
    W = df.FunctionSpace(M, df.MixedElement([VE, RE1, RE2]), constrained_domain = PeriodicBoundary() )   

    corner = df.CompiledSubDomain("(std::abs(x[0]) < 0.0001) && (std::abs(x[1]) < 0.0001)")
    points = df.MeshFunction('size_t', M, 0)
    points.set_all(0)
    corner.mark(points,1)
    dp = df.Measure('dP',subdomain_data =points)
    
    u1,u2,p1,p2,l1,l2 = df.TrialFunction(W)
    v1,v2,q1,q2,m1,m2 = df.TestFunction(W)
    
    p = df.as_vector((p1,p2))
    q = df.as_vector((q1,q2))
    u = df.as_vector((u1,u2))
    v = df.as_vector((v1,v2))
    l = df.as_vector((l1,l2))
    m = df.as_vector((m1,m2))

    a = df.inner(sigma(u),fela.epsilon(v))*M.dx -  df.inner(p,v)*M.dx - df.inner(q,u)*M.dx
    a += df.inner(m,u)*dp(1) + df.inner(l,v)*dp(1)
    L = -df.inner(sigmaEps, fela.epsilon(v))*M.dx    
 
    return a, L, [], W


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

def homogenisation(u, M, sigma, domains, sigmaEps = df.Constant(((0.0,0.0),(0.0,0.0)))):
    s = sigma(u)
    sigma_hom = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            sij = s[i,j] + sigmaEps[i,j]
            sigma_hom[i,j] = sum( [df.assemble(sij*M.dx(k)) for k in domains]) 

    vol = sum( [df.assemble(df.Constant(1.0)*M.dx(k)) for k in domains]) 
    
    sigma_hom = sigma_hom/vol

    return sigma_hom

class PeriodicBoundary(df.SubDomain):
    # Left boundary is "target domain" G
    def __init__(self,x0 = 0.0,x1 = 1.0,y0 = 0.0 ,y1 = 1.0, **kwargs):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        
        super().__init__(**kwargs)
    
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
        if(on_boundary):
            left, bottom, right, top = self.checkPosition(x)
            return (left and not top) or (bottom and not right)
        
        return False
     
    def checkPosition(self,x):
        return df.near(x[0], self.x0), df.near(x[1],self.y0), df.near(x[0], self.x1), df.near(x[1], self.y1)
    
    def map(self, x, y):
        left, bottom, right, top = self.checkPosition(x)
        
        y[0] = x[0] + self.x0 - (self.x1 if right else self.x0)
        y[1] = x[1] + self.y0 - (self.y1 if top else self.y0)

class PeriodicBoundaryInverse(df.SubDomain):
    # Left boundary is "target domain" G
    def __init__(self,x0 = 0.0,x1 = 1.0,y0 = 0.0 ,y1 = 1.0, **kwargs):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        
        super().__init__(**kwargs)
    
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
        if(on_boundary):
            left, bottom, right, top = self.checkPosition(x)
            return (right and not bottom) or (top and not left)
        
        return False
     
    def checkPosition(self,x):
        return df.near(x[0], self.x0), df.near(x[1],self.y0), df.near(x[0], self.x1), df.near(x[1], self.y1)
    
    def map(self, x, y):
        left, bottom, right, top = self.checkPosition(x)
        
        y[0] = x[0] + (self.x1 - self.x0 if left else 0.0)
        y[1] = x[1] + (self.y1 - self.y0 if bottom else 0.0)
    

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


def getAntiperiodic(f, r = df.Expression(('2.*x0 - x[0]','2.*y0 - x[1]'), x0 = 0.5,  y0 = 0.5, degree = 1) ):
    fr = df.interpolate(feut.myfog(f,r), f.function_space())
    fr.vector().set_local( 0.5*(f.vector()[:] - fr.vector()[:]))
    return fr


