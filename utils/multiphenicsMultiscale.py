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
import multiphenics as mp
from fenicsMultiscale import getSigma_SigmaEps, homogenisedTangent, getTotalDisplacement, homogenisation, PointExpression, getSigmaEps, getSigma, PeriodicBoundary

def solveMultiscale(param, M, eps, op, others = {}):

    sigma, sigmaEps = getSigma_SigmaEps(param,M,eps, op = 'cpp')    
        
    if(op == 'MR'):
        a,f,bcs,W = formulationMultiscaleMR(M, sigma, sigmaEps)
    elif(op == 'periodic'):
        x0 = others['per'][0]
        x1 = others['per'][1]
        y0 = others['per'][2]
        y1 = others['per'][3]
            
        a,f,bcs,W = formulationMultiscale_periodic(M, sigma, sigmaEps, x0, x1, y0, y1)
        
    elif(op == 'POD'):
        bbasis = others[0] 
        alpha = others[1] 
        a,f,bcs,W = formulationMultiscale_POD(M, sigma, sigmaEps, bbasis, alpha)
    elif(op == 'POD_noMR'):
        bbasis = others[0] 
        alpha = others[1] 
        a,f,bcs,W = formulationMultiscale_POD_noMR(M, sigma, sigmaEps, bbasis, alpha)
    elif(op == 'BCdirich'):
        linBdr = others[0]
        uD = others[1]
        
        if(type(uD) == type(np.zeros(1))):
            g = gmts.displacementGeneratorBoundary(0.0,0.0,1.0,1.0, int((uD_.shape[0] + 8)/8) )
            uD = PointExpression(uD_, g)

        a,f,bcs,W = formulationMultiscaleBCdirich_zeroMean(M, sigma, sigmaEps, linBdr, uD)
                
    elif(op == 'BCdirich_lag'):
        uD = others[0]

        if(type(uD) == type(np.zeros(1))):
            g = gmts.displacementGeneratorBoundary(0.0,0.0,1.0,1.0, int((uD_.shape[0] + 8)/8) )
            uD = PointExpression(uD_, g)

        a,f,bcs,W = formulationMultiscaleBCdirich_lag_zeroMean(M, sigma, sigmaEps, uD)
        
    elif(op == 'Lin'):
        bdr = others['bdr']
        a,f,bcs,W = formulationMultiscaleBClinear(M, sigma, sigmaEps, bdr = bdr)
    
    else:
        print('option', op, 'havent been found')
        
    start = timer()
    A = mp.block_assemble(a)
    F = mp.block_assemble(f)
    
    if(len(bcs) > 0):
        bcs.apply(A)
        bcs.apply(F)
    
    # condNumber = np.linalg.cond(A.array())
    
    sol = mp.BlockFunction(W)
    mp.block_solve(A, sol.block_vector(), F, 'mumps')

    end = timer()
    print('time in solving system', end - start) # Time in seconds, e.g. 5.38091952400282

    # f = open('conditionNumber.txt', 'a')
    # print('condition number is ', condNumber ) # Time in seconds, e.g. 5.38091952400282
    # f.write(str(condNumber) + '\n')

    # f.close()
    return sol

def formulationMultiscaleMR(M, sigma, sigmaEps):

    V = df.VectorFunctionSpace(M,"CG", 1)
    R1 = df.VectorFunctionSpace(M, "Real", 0)
    R2 = df.TensorFunctionSpace(M, "Real", 0)
    
    W = mp.BlockFunctionSpace([V,R1,R2])   
    
    # Define variational problem
    
    uu = mp.BlockTrialFunction(W)
    vv = mp.BlockTestFunction(W)
    (u, p, P) = mp.block_split(uu)
    (v, q, Q) = mp.block_split(vv)
    
    n = df.FacetNormal(M)
    
    # Create the block matrix for the block LHS
    aa = []
    aa.append([df.inner(sigma(u),fela.epsilon(v))*M.dx, - df.inner(p,v)*M.dx, - df.inner(P,df.outer(v,n))*M.ds])
    aa.append([- df.inner(q,u)*M.dx, 0, 0]), 
    aa.append([ - df.inner(Q,df.outer(u,n))*M.ds, 0, 0])
    
    ff = [-df.inner(sigmaEps, fela.epsilon(v))*M.dx, 0, 0]
    
    return aa, ff, [], W 


def formulationMultiscaleBCdirich_zeroMean(M, sigma, sigmaEps, linBdr, uD):

    V = df.VectorFunctionSpace(M,"CG", 1)
    R = df.VectorFunctionSpace(M, "Real", 0)
    
    W = mp.BlockFunctionSpace([V,R])   
    
    uD = df.Function(V)
    uD.vector().set_local(np.zeros(V.dim()))
    
    bc1 = mp.DirichletBC(W.sub(0), uD , M.boundaries, linBdr[0]) 
                
    uu = mp.BlockTrialFunction(W)
    vv = mp.BlockTestFunction(W)
    (u, p) = mp.block_split(uu)
    (v, q) = mp.block_split(vv)

    aa = [[df.inner(sigma(u),fela.epsilon(v))*M.dx , df.inner(p,v)*M.dx], [df.inner(q,u)*M.dx , 0]]
    ff = [-df.inner(sigmaEps, fela.epsilon(v))*M.dx, 0]    
 
    
    bcs = mp.BlockDirichletBC([bc1])
    
    return aa, ff, bcs, W



def formulationMultiscaleBClinear(M, sigma, sigmaEps, bdr):

    V = df.VectorFunctionSpace(M,"CG", 1)
    R = df.VectorFunctionSpace(M, "Real", 0)
    
    W = mp.BlockFunctionSpace([V,R])   
    
    bc1 = mp.DirichletBC(W.sub(0), df.Constant((0.,0.)) , M.boundaries, bdr) 
                
    uu = mp.BlockTrialFunction(W)
    vv = mp.BlockTestFunction(W)
    (u, p) = mp.block_split(uu)
    (v, q) = mp.block_split(vv)

    aa = [[df.inner(sigma(u),fela.epsilon(v))*M.dx , df.inner(p,v)*M.dx], [df.inner(q,u)*M.dx , 0]]
    ff = [-df.inner(sigmaEps, fela.epsilon(v))*M.dx, 0]    
    
    bcs = mp.BlockDirichletBC([bc1])
    
    return aa, ff, bcs, W

def formulationMultiscaleBCdirich_lag_zeroMean(M, sigma, sigmaEps, uD):

    if('u' in M.V.keys()):
        print("using  ", M.V['u'])
        V = M.V['u']
    else:
        V = df.VectorFunctionSpace(M,"CG", 1)
    
    R = df.VectorFunctionSpace(M, "Real", 0)
    
    class OnBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    onBoundary = OnBoundary()
    bmesh = mp.MeshRestriction(M, onBoundary)
    
    W = mp.BlockFunctionSpace([V,V,R] , restrict = [None, bmesh, None])   
                
    ulp = mp.BlockTrialFunction(W)
    vmq = mp.BlockTestFunction(W)
    (u, l, p) = mp.block_split(ulp)
    (v, m, q) = mp.block_split(vmq)

    aa = []
    aa.append([df.inner(sigma(u),fela.epsilon(v))*M.dx , df.inner(l,v)*M.ds, df.inner(p,v)*M.dx])
    aa.append([df.inner(m,u)*M.ds, 0, 0])
    aa.append([df.inner(q,u)*M.dx , 0, 0])
    
    ff = [-df.inner(sigmaEps, fela.epsilon(v))*M.dx, df.inner(m,uD)*M.ds, 0]    
 
    
    return aa, ff, [], W

def formulationMultiscale_periodic(M, sigma, sigmaEps, x0 = 0., x1 = 1., y0 = 0., y1 = 1.):

    V = df.VectorFunctionSpace(M,"CG", 1, constrained_domain = PeriodicBoundary(x0,x1,y0,y1))
    R = df.VectorFunctionSpace(M, "Real", 0)
    
    W = mp.BlockFunctionSpace([V,R])   
    
    uu = mp.BlockTrialFunction(W)
    vv = mp.BlockTestFunction(W)
    (u, p) = mp.block_split(uu)
    (v, q) = mp.block_split(vv)

    aa = [[df.inner(sigma(u),fela.epsilon(v))*M.dx , df.inner(p,v)*M.dx], [df.inner(q,u)*M.dx , 0]]
    ff = [-df.inner(sigmaEps, fela.epsilon(v))*M.dx, 0]    
 
    return aa, ff, [], W

def formulationMultiscale_POD(M, sigma, sigmaEps, bbasis, alpha):
    m = len(bbasis)
    
    V = df.VectorFunctionSpace(M,"CG", 1)    
    RV = df.VectorFunctionSpace(M, "Real", 0)
    RT = df.TensorFunctionSpace(M, "Real", 0)
    Rlag = df.VectorFunctionSpace(M, "Real", dim = m, degree = 0)
    
    W = mp.BlockFunctionSpace([V,RV,RT,Rlag])   
    
    upPr = mp.BlockTrialFunction(W)
    vqQs = mp.BlockTestFunction(W)
    u,p,P,r = mp.block_split(upPr)
    v,q,Q,s = mp.block_split(vqQs)

    n = df.FacetNormal(M)
    
    # Create the block matrix for the block LHS
    aa = []
    aa.append([df.inner(sigma(u),fela.epsilon(v))*M.dx, - df.inner(p,v)*M.dx, - df.inner(P,df.outer(v,n))*M.ds])
    aa.append([ -df.inner(q,u)*M.dx, 0, 0]) 
    aa.append([ -df.inner(Q,df.outer(u,n))*M.ds, 0, 0])

    ff = [-df.inner(sigmaEps, fela.epsilon(v))*M.dx , 0 , 0]     
    
    for i in range(m):
        aa[0].append(r[i]*df.inner(bbasis[i],v)*M.ds)
        aa[1].append(0)
        aa[2].append(0)
        aa.append([s[i]*df.inner(bbasis[i],u)*M.ds] + (m+2)*[0])
        ff.append(s[i]*alpha[i]*M.ds)
      
    
    return aa, ff, [], W



def formulationMultiscale_POD_noMR(M, sigma, sigmaEps, bbasis, alpha):
    m = len(bbasis)
    
    V = df.VectorFunctionSpace(M,"CG", 1)    
    RV = df.VectorFunctionSpace(M, "Real", 0)
    Rlag = df.VectorFunctionSpace(M, "Real", dim = m, degree = 0)
    
    W = mp.BlockFunctionSpace([V,RV,Rlag])   
    
    upr = mp.BlockTrialFunction(W)
    vqs = mp.BlockTestFunction(W)
    u,p,r = mp.block_split(upr)
    v,q,s = mp.block_split(vqs)

    # Create the block matrix for the block LHS
    aa = []
    aa.append([df.inner(sigma(u),fela.epsilon(v))*M.dx, - df.inner(p,v)*M.dx])
    aa.append([ -df.inner(q,u)*M.dx, 0])
    
    ff = [-df.inner(sigmaEps, fela.epsilon(v))*M.dx , 0 ]     
    
    for i in range(m):
        aa[0].append(r[i]*df.inner(bbasis[i],v)*M.ds)
        aa[1].append(0)
        aa.append([s[i]*df.inner(bbasis[i],u)*M.ds] + (m+1)*[0])
        ff.append(s[i]*alpha[i]*M.ds)
      
    
    return aa, ff, [], W

def projectOnBoundary(ustar): # as far as remember, it crashes
    V = ustar.function_space()
    M = V.mesh()
        
    class OnBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    onBoundary = OnBoundary()
    bmesh = mp.MeshRestriction(M, onBoundary)
    
    W = mp.BlockFunctionSpace([V] , restrict = [bmesh])   

    u = mp.BlockTrialFunction(W)
    v = mp.BlockTestFunction(W)
    
    u = mp.block_split(u)[0]
    v = mp.block_split(v)[0]
    
    
    aa = [[df.inner(u,v)*M.ds]]
    ff = [df.inner(ustar,v)*M.ds]
    
    A = mp.block_assemble(aa)
    F = mp.block_assemble(ff)
    
    sol = mp.BlockFunction(W)
    mp.block_solve(A, sol.block_vector(), F)
    
    return sol

    