import sys, os
sys.path.insert(0, '../../utils/')

import matplotlib.pyplot as plt
import numpy as np
import generationInclusions as geni
import myHDF5 as myhd
from timeit import default_timer as timer
from scipy.spatial.distance import cdist



def gaussian_correlation(x,sigma,l):
    n = len(x)
    dist =  cdist(x, x, metric='euclidean')   
    return sigma**2 * np.exp(- dist**2/l**2)



Lx = Ly = 6. # length of the edge of the squared domain
dim = 2 # the domain is a dim-dimensional square
nx = ny = 50

hx = Lx/float(nx) # mesh size
hy = Ly/float(ny)

# stochastic parameters
sigma = 1.0
mu = 0.0
corr_length = 0.3

x = np.linspace(-Lx/2,Lx/2,nx + 1)
y = np.linspace(-Ly/2,Ly/2,ny + 1)

meshXY  = np.meshgrid(x,y)
nx = meshXY[0].shape[0]
ny = meshXY[0].shape[1]
n = nx*ny
meshXY = np.concatenate( (meshXY[0].reshape((n,1)) , meshXY[1].reshape((n,1))) , axis = 1 )


start = timer()
cov = gaussian_correlation(meshXY, sigma, corr_length)
end = timer()
print(end - start)

mean = mu*np.ones(n)

Ns = 4

start = timer()
I = np.random.multivariate_normal(mean, cov, size = (Ns)).reshape((Ns,nx,ny))

end = timer()
print(end - start)

# zeros(size(mesh.node,1),1),Cov

minI = np.floor(np.min(I))
maxI = np.ceil(np.max(I))
rangeI = maxI - minI
    
for i in range(Ns):    
    plt.figure(i,(4,4))
    plt.imshow(I[i,:,:],interpolation="bicubic")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("fig_" + str(i) + ".png")
    # plt.colorbar(ticks=list(np.linspace(minI, maxI, rangeI + 1)))