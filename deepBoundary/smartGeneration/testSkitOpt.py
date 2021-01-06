import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt
from skopt.space import Space
from skopt.sampler import Sobol
from skopt.sampler import Lhs
from skopt.sampler import Halton
from skopt.sampler import Hammersly
from skopt.sampler import Grid
from scipy.spatial.distance import pdist

def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]

print(numberToBase(45,3))

def getScikitoptSampleVolFraction_adaptativeSample(NR,Vfrac,r0,r1, p, rm, facL, facR, H, M, N, 
                                                   indexes, seed, op = 'lhs'):
    space = Space(NR*[(r0, r1)])
    if(op == 'lhs'):
        sampler = Lhs(lhs_type="centered", criterion=None)
    elif(op == 'lhs_maxmin'):
        sampler = Lhs(criterion="maximin", iterations=100)
    elif(op == 'sobol'):
        sampler = Sobol()
    
    Rlist = []
    np.random.seed(seed)
    
    for pi in range(p**M):
        pibin = numberToBase(pi,p) # supposing p = 2
        pibin = pibin + (4-len(pibin))*[0] # to complete 4 digits
        
        Rlist.append( np.array(sampler.generate(space.dimensions, N)) )
        
        for j, jj in enumerate(indexes):
            k = pibin[j]
            if(k == 0):
                Rlist[-1][:,jj] = r0 + facL*( Rlist[-1][:,jj] - r0 )
            elif(k == 1):
                Rlist[-1][:,jj] = rm + facR*( Rlist[-1][:,jj] - r0 )
    
        R = np.concatenate(Rlist,axis = 0)
    
    # for i in range(N*p**M): # total samples, impose volume fraction in the whole volume (maybe it should be partionated)
    #     alphaFrac = H*np.sqrt(NR*Vfrac/(np.pi*np.sum(R[i,:]**2)))
    #     R[i,:] *= alphaFrac
    
    return R


def getScikitoptSampleVolFraction_adaptativeSample_frozen(NR,Vfrac,r0,r1, H, p, M, N, 
                                                   indexes, seed, op = 'lhs'):
    space = Space(NR*[(r0, r1)])
    if(op == 'lhs'):
        sampler = Lhs(lhs_type="centered", criterion=None)
    elif(op == 'lhs_maxmin'):
        sampler = Lhs(criterion="maximin", iterations=100)
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
    
    # for i in range(N*p**M): # total samples, impose volume fraction in the whole volume (maybe it should be partionated)
    #     alphaFrac = H*np.sqrt(NR*Vfrac/(np.pi*np.sum(R[i,:]**2)))
    #     R[i,:] = alphaFrac*R[i,:]
    
    return R


def plot_searchspace(x, title):
    fig, ax = plt.subplots()
    plt.plot(np.array(x)[:, 0], np.array(x)[:, 1], 'bo', label='samples')
    plt.plot(np.array(x)[:, 0], np.array(x)[:, 1], 'bo', markersize=80, alpha=0.5)
    # ax.legend(loc="best", numpoints=1)
    ax.set_xlabel("X1")
    ax.set_xlim([-5, 10])
    ax.set_ylabel("X2")
    ax.set_ylim([0, 15])
    plt.title(title)

n_samples = 24

l1 = 0.0
l2 = 00.0
r1 = 80.0
r2 = 40.0
m1 = 0.5*(l1 + r1) 
m2 = 0.5*(l2 + r2) 
fac1L = (m1 - l1)/(r1 - l1)
fac1R = (r1 - m1)/(r1 - l1)
fac2L = (m2 - l2)/(r2 - l2)
fac2R = (r2 - m2)/(r2 - l2)

p = 3
M = 1
N = int(n_samples/(p**M))

space = Space([(l1, r1), (l2, r2)])
# space.set_transformer("normalize")

np.random.seed(5)
x = np.array(space.rvs(n_samples))
plt.figure(1,(6,6))
# plt.title("Random samples")
# plt.scatter(x[:,0],x[:,1], label = 'random')
# pdist_data = []
# x_label = []
# pdist_data.append(pdist(x).flatten())
# x_label.append("random")

np.random.seed(7)
lhs = Lhs(lhs_type="centered", criterion=None)
x0 = np.array(lhs.generate(space.dimensions, n_samples))

lhs2 = Lhs(criterion="maximin", iterations=10000)
x2 = np.array(lhs2.generate(space.dimensions, n_samples))

sobol = Sobol()
x1 = np.array(sobol.generate(space.dimensions, n_samples))



# plt.figure(1)
# plt.title('centered LHS')
# plt.scatter(x0[:,0],x[:,1], label = 'LHS')
# plt.scatter(x2[:,0],x[:,1], label = 'LHS maxmin')
# plt.scatter(x1[:,0],x[:,1], label = 'Sobol')
# plot_searchspace(x, )
# pdist_data.append(pdist(x).flatten())
# x_label.append("center")

r0 = 0.2
r1 = 0.6
rm = 0.5*(r0 + r1) 
facL = (rm - r0)/(r1 - r0)
facR = (r1 - rm)/(r1 - r0)
H = 0.1

# rlim = [r0,rm]
# faclim = [facL,facR]

# xnew = getScikitoptSampleVolFraction_adaptativeSample_frozen(2,0.2,rlim, faclim, H, p, M, N, 
#                                                    [1], seed = 1 , op = 'lhs_maxmin')


rlim = [r0 + i*(r1-r0)/p for i in range(p+1)]
faclim = [(rlim[i+1]-rlim[i])/(r1-r0) for i in range(p)]

xnew = getScikitoptSampleVolFraction_adaptativeSample_frozen(2,0.2,rlim, faclim, H, p, M, N, [0], seed = 7, op = 'lhs_maxmin')


# xlist = []
# np.random.seed(7)
# lhs = Lhs(criterion="maximin", iterations = 100)
# for pi in range(p):
#     xlist.append( np.array(lhs.generate(space.dimensions, N)) )
#     print(xlist[-1][:,0])
#     if(pi == 0):
#         xlist[-1][:,0] = l1 + fac1L*( xlist[-1][:,0] - l1 )
#     elif(pi == 1):
#         xlist[-1][:,0] = m1 + fac1R*( xlist[-1][:,0] - l1 )

# xnew = np.concatenate(xlist,axis = 0)

plt.scatter(xnew[:8,0],xnew[:8,1], label = 'LHS 1')
plt.scatter(xnew[8:16,0],xnew[8:16,1], label = 'LHS 2')
plt.scatter(xnew[16:,0],xnew[16:,1], label = 'LHS 3')
plt.grid()

plt.legend(loc = 'best')


