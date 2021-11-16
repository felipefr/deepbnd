def testOrthonormalityBasis(NrandTests , Wbasis, Vref, dxRef, dotProduct, Nmax = -1, silent = False):
    
    if(Nmax < 0):
        Nmax = len(Wbasis)
        
    wi = Function(Vref)
    wj = Function(Vref)
    
    target = np.ones(3*NrandTests)
    target[0::3] = 0.0
    prods = np.zeros(3*NrandTests)
    
    for k in range(NrandTests):
        i = np.random.randint(0,Nmax)
        j = np.random.randint(0,Nmax)
        if(i==j):
            j = i + 1
        
        wi.vector().set_local(Wbasis[i,:])
        wj.vector().set_local(Wbasis[j,:])               
        
        prods[3*k] = dotProduct(wi,wj,dxRef)
        prods[3*k + 1] = dotProduct(wi,wi,dxRef)
        prods[3*k + 2] = dotProduct(wj,wj,dxRef)
        
        if(not silent):
            print('(w{0},w{1}) = {2}'.format(i,j,prods[3*k]))
            print('(w{0},w{1}) = {2}'.format(i,i,prods[3*k + 1]))
            print('(w{0},w{1}) = {2}'.format(j,j,prods[3*k + 2]))
    
                                             
    return np.linalg.norm(prods - target)/(3*NrandTests)