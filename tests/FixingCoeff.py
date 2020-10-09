import sys
sys.path.insert(0, '../utils/')

import matplotlib.pyplot as plt
import numpy as np
import multiphenics as mp
import dolfin as df
import meshUtils as meut
import elasticity_utils as elut
import myCoeffClass as coef
import copy
from timeit import default_timer as timer

epsilon = lambda u: 0.5*(df.nabla_grad(u) + df.nabla_grad(u).T)
eng2lame = lambda p : np.array([ [elut.eng2lambPlane(pi[0], pi[1]), elut.eng2mu(pi[0], pi[1])] for pi in p]) 

ellipseData = np.array([[0.5,0.5,0.3,1.0,0.0]]) # x0,y0,r,excentricity, theta

meshGMSH = meut.ellipseMesh2(ellipseData, x0 = 0.0, y0 = 0.0, Lx = 1.0 , Ly = 1.0 , lcar = 0.01)
meshGMSH.setTransfiniteBoundary(100)
meshGMSH.setNameMesh('basicMesh.xdmf')
M = meshGMSH.getEnrichedMesh() 

nu = 0.3
E = 10.0
contrast = 0.2
param = eng2lame([[nu,E],[nu,contrast*E]])

def getSigma(M,param):
    lame =  coef.getMyCoeff(M.subdomains.array(), param, 'cpp')    
    sigma = lambda eps: lame[0]*df.tr(eps)*df.Identity(2) + 2*lame[1]*eps
    return sigma

eps = np.array([[1.0,0.0],[0.0,0.0]])
Eps = df.Constant(0.5*(eps + eps.T))
start = timer()
   
sigma = getSigma(M,param)

# Multiphenics version
V = df.VectorFunctionSpace(M,"CG", 1)
R = df.VectorFunctionSpace(M, "Real", 0)

W = mp.BlockFunctionSpace([V,R])   

uu = mp.BlockTrialFunction(W)
vv = mp.BlockTestFunction(W)
(u, p) = mp.block_split(uu)
(v, q) = mp.block_split(vv)

aa = [[df.inner(sigma(epsilon(u)),epsilon(v))*M.dx , df.inner(p,v)*M.dx], [df.inner(q,u)*M.dx , 0]]
ff = [-df.inner(sigma(Eps), epsilon(v))*M.dx, 0]  

bc1 = mp.DirichletBC(W.sub(0), df.Constant((0.,0.)) , M.boundaries, 2) 
bcs = mp.BlockDirichletBC([bc1])

A = mp.block_assemble(aa)
F = mp.block_assemble(ff)


bcs.apply(A)
bcs.apply(F)

sol = mp.BlockFunction(W)
mp.block_solve(A, sol.block_vector(), F, 'mumps')

end = timer()
print(end - start)

u = sol[0]
p = sol[1]

plt.figure(1)
df.plot(u[0])
plt.figure(2)
df.plot(u[1])
plt.figure(3)
df.plot(p[0])
