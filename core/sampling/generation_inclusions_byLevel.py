import numpy as np
from skopt.space import Space
from skopt.sampler import Lhs, Sobol


def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]

def getScikitoptSample_LHSbyLevels(NR,ns,p,indexes,r0,r1, seed, op = 'lhs'):
    M = len(indexes)
    N = int(ns/(p**M))
    
    space = Space(NR*[(r0, r1)])
    if(op == 'lhs'):
        sampler = Lhs(lhs_type="centered", criterion=None)
    elif(op == 'lhs_maxmin'):
        sampler = Lhs(criterion="maximin", iterations=20)
    elif(op == 'sobol'):
        sampler = Sobol()
    
    Rlist = []
    np.random.seed(seed)
    
        
    rlim = [r0 + i*(r1-r0)/p for i in range(p+1)]
    faclim = [(rlim[i+1]-rlim[i])/(r1-r0) for i in range(p)]
    
    for pi in range(p**M):
        pibin = numberToBase(pi,p) # supposing p = 2
        pibin = pibin + (4-len(pibin))*[0] # to complete 4 digits
        
        Rlist.append( np.array(sampler.generate(space.dimensions, N)) )
        
        for j in range(len(Rlist[-1][0,:])):
            if(j not in indexes):
                Rlist[-1][:,j] = Rlist[0][:,j]
        
        for j, jj in enumerate(indexes): #j = 0,1,2,..., jj = I_0,I_1,I_2,...
            k = pibin[j]
            Rlist[-1][:,jj] = rlim[k] + faclim[k]*( Rlist[-1][:,jj] - r0 )
            
    
        R = np.concatenate(Rlist,axis = 0)
    

    return R