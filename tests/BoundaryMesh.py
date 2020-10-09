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
import fenicsUtils as feut

epsilon = lambda u: 0.5*(df.nabla_grad(u) + df.nabla_grad(u).T)
eng2lame = lambda p : np.array([ [elut.eng2lambPlane(pi[0], pi[1]), elut.eng2mu(pi[0], pi[1])] for pi in p]) 

ellipseData = np.array([[0.5,0.5,0.3,1.0,0.0]]) # x0,y0,r,excentricity, theta

meshGMSH = meut.ellipseMesh2(ellipseData, x0 = 0.0, y0 = 0.0, Lx = 1.0 , Ly = 1.0 , lcar = 0.01)
meshGMSH.setTransfiniteBoundary(100)
# M = meut.getMesh(meshGMSH, meshXmlFile = 'basicMesh.xml', create = 'True')    
meshGMSH.generate(gmsh_opt = ['-algo','del2d'])
meshGMSH.write('basicMesh.xdmf', 'fenics')
M = meut.EnrichedMesh('basicMesh.xdmf')

meshGMSH2 = meut.degeneratedBoundaryRectangleMesh(x0 = 0.0, y0 = 0.0, Lx = 1.0 , Ly = 1.0 , Nb = 100)
meshGMSH2.generate()
meshGMSH2.write('boundaryMesh.xdmf', 'fenics')
M2 = meut.EnrichedMesh('boundaryMesh.xdmf')

# M2 = meut.getMesh(meshGMSH2, meshXmlFile = 'boundaryMesh.xml', create = 'True')  


# 'gmsh -2 -algo del2d -format msh2 basicMesh.geo' 4.6.0


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
R1 = df.VectorFunctionSpace(M, "Real", 0)
R2 = df.TensorFunctionSpace(M, "Real", 0)

W = mp.BlockFunctionSpace([V,R1,R2])   

uu = mp.BlockTrialFunction(W)
vv = mp.BlockTestFunction(W)
(u, p, P) = mp.block_split(uu)
(v, q, Q) = mp.block_split(vv)

n = df.FacetNormal(M)


# Define variational problem

aa = []
aa.append([df.inner(sigma(epsilon(u)),epsilon(v))*M.dx, - df.inner(p,v)*M.dx, - df.inner(P,df.outer(v,n))*M.ds])
aa.append([- df.inner(q,u)*M.dx, 0, 0]), 
aa.append([ - df.inner(Q,df.outer(u,n))*M.ds, 0, 0])

ff = [-df.inner(sigma(Eps), epsilon(v))*M.dx, 0, 0]

A = mp.block_assemble(aa)
F = mp.block_assemble(ff)

sol = mp.BlockFunction(W)
mp.block_solve(A, sol.block_vector(), F, 'mumps')

end = timer()
print(end - start)

u = sol[0]
p = sol[1]

print(feut.Integral(df.outer(u,n),M.ds,shape = (2,2)))

V2 = df.VectorFunctionSpace(M2,"CG", 1)
u2 = df.interpolate(u,V2)
n2 = df.FacetNormal(M2)

print(feut.Integral(df.outer(u2,n2),M2.ds,shape = (2,2)))

print(df.assemble(df.inner(u,u)*M.ds))
print(df.assemble(df.inner(u2,u2)*M2.ds))

plt.figure(1)
df.plot(u[0])
plt.figure(2)
df.plot(u[1])
plt.figure(3)
df.plot(p[0])
