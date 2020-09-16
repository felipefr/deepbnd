import dolfin as df
import fenicsWrapperElasticity as fela

def getAffineDecomposition(M):
    
    M.addDirichletBC('clamped','u', df.Constant((0.0,0.0)) , 1)
    M.nameNeumannBoundary('Right_1',2)
    M.nameNeumannBoundary('Right_2',3)
    M.nameNeumannBoundary('Right_3',4)
   
    # Define variational problem
    u = df.TrialFunction(M.V['u'])
    v = df.TestFunction(M.V['u'])
    
    a = []
    a.append(df.inner(u[0].df.dx(0), v[0].df.dx(0)))
    a.append(df.inner(u[1].df.dx(1), v[1].df.dx(1)))
    a.append(df.inner(u[1].df.dx(0), v[1].df.dx(0)))
    a.append(df.inner(u[0].df.dx(1), v[0].df.dx(1)))
    a.append(df.inner(u[1].df.dx(1), v[0].df.dx(0)) + df.inner(u[0].df.dx(0), v[1].df.dx(1)))
    a.append(df.inner(u[0].df.dx(1), v[1].df.dx(0)) + df.inner(u[1].df.dx(0), v[0].df.dx(1)))

    
    A = []
    
    for i in range(10):
        for aa in a:
             # used to be abble to apply boundary conditions in the generation of affine decomposition terms
            A.append(df.assemble(aa*M.df.dx(i) , keep_diagonal = True))
            
    T = df.Constant((-0.2,  0.0 ))
    L = df.dot(T, v)
    
    F = []
    
    for dsN in M.dsN.values():
        F.append(df.assemble(L*dsN))
        
    thetasA = []
    thetasA.append(lambda lamb,mu,l1,l2: (lamb + 2.*mu)*l2/l1)
    thetasA.append(lambda lamb,mu,l1,l2: (lamb + 2.*mu)*l1/l2)
    thetasA.append(lambda lamb,mu,l1,l2: mu*l2/l1)
    thetasA.append(lambda lamb,mu,l1,l2: mu*l1/l2)
    thetasA.append(lambda lamb,mu,l1,l2: lamb)
    thetasA.append(lambda lamb,mu,l1,l2: mu)
    
    thetasF = [lambda lamb,mu,l1,l2: l2]

    for Ai in A:
        M.applyDirichletBCs(Ai)
       
    return A, thetasA, F, thetasF

def solveElasticityBimaterial_withAffDec(param, M, paramModMesh):
    
    M.createFiniteSpace(spaceType = 'V', name = 'u', spaceFamily = 'CG', degree = 2)
    
    Alist, thetasA, Flist, thetasF = getAffineDecomposition(M)
    d = fela.getDefaultParameters()
    b = fela.getAffineParameters(paramModMesh, d)[1]
    
    F = df.Vector(Flist[0])
    F.zero()
    A = df.Matrix(Alist[0])
    A.zero()
    
    listDomainPosition = [(0,0),(1,0),(2,0),(0,1), (2,1), (0,2), (1,2), (2,2), (1,1), (1,1)]
    for i in range(10):
        lamb, mu = param[0] if i<9 else param[1]
        ii,jj = listDomainPosition[i]
        l1 = b[ii,0]
        l2 = b[jj,1]

        for j, theta in enumerate(thetasA):
            A+=theta(lamb,mu,l1,l2)*Alist[i*6 + j]
    

    for i in range(3):
        l2 = b[i,1]
        
        for theta in thetasF:
            F+= theta(0.0,0.0,0.0,l2)*Flist[i]       
    
    u = df.Function(M.V['u'])
    df.solve(A,u.vector(),F)
        
    return u

# ===========================  this may be useful ========================================================= 
#     K = [assemble(a1), assemble(a2)]
#     rhs = [assemble(L1) , assemble(L2)]
    
#     bc.apply(K[0],rhs[0])
#     bc.apply(K[1],rhs[1])
    
#     VbaseT = Vbase.transpose()
#     affineDecomposition = {'ANq0':0, 'ANq1':0, 'fNq0':0, 'fNq1':0, 'Aq0':0, 'Aq1':0, 'fq0':0, 'fq1':0}
#     affineDecomposition['ANq0'] = np.dot(np.dot(VbaseT,K[0].array()), Vbase)
#     affineDecomposition['ANq1'] = np.dot(np.dot(VbaseT,K[1].array()), Vbase)
#     affineDecomposition['fNq0'] = np.dot(VbaseT,rhs[0].get_local())
#     affineDecomposition['fNq1'] = np.dot(VbaseT,rhs[1].get_local())
    
#     affineDecomposition['Aq0'] = K[0].array()
#     affineDecomposition['Aq1'] = K[1].array()
#     affineDecomposition['fq0'] = rhs[0].get_local()
#     affineDecomposition['fq1'] = rhs[0].get_local()
    
#     return affineDecomposition

# def computeRBapprox(param,affineDecomposition,Vbase):
    
#     AN = np.zeros((N,N))
#     for j in range(2):
#         AN += param[j]*affineDecomposition['ANq' + str(j)]

#     fN = np.zeros(N)
#     theta_f = [np.cos(param[2]), np.sin(param[2])]
#     for j in range(2):
#         fN += theta_f[j]*affineDecomposition['fNq' + str(j)] 
    
#     uN = np.linalg.solve(AN,fN)
    

#     uR=np.dot(Vbase,uN).flatten() 

    
#     return uR
