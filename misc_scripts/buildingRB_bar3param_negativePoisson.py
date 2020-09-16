from __future__ import print_function
import numpy as np
from fenics import *
from dolfin import *
from ufl import nabla_div
import matplotlib.pyplot as plt

import elasticity_utils as elut

factorForce_y = 0.1
bodyForce = lambda theta: (np.cos(theta),factorForce_y*np.sin(theta)) 

def getFEspace():
    # Scaled variables
    L = 5.0; W = 1.0
   # Create mesh and define function space
    mesh = RectangleMesh(Point(0, -W/2.), Point(L, W/2.), 50, 10)
    
    V = VectorFunctionSpace(mesh, 'CG', 1)
    
    return V


def solveElasticityBarFenics(V,param, isLame = False, savevtk = None):


    # Define boundary condition
    tol = 1E-14
    
    def clamped_boundary(x, on_boundary):
        return on_boundary and x[0] < tol
    
    bc = DirichletBC(V, Constant((0, 0)), clamped_boundary)
    
    # Define strain and stress

    lamb = Constant(0.0)
    mu = Constant(0.0)
    
    def epsilon(u):
        return 0.5*(nabla_grad(u) + nabla_grad(u).T)
    
    def sigma(u):
        return lamb*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)
    
    # Define variational problem
    u = TrialFunction(V)
    d = u.geometric_dimension()  # space dimension
    v = TestFunction(V)
    f = Constant((0, 0))
    T = Constant((0, 0))
    a = inner(sigma(u), epsilon(v))*dx
    L = dot(f, v)*dx + dot(T, v)*ds
    
    
    if(isLame):
        lamb_value = param[0]
        mu_value = param[1]
    else:
        E = param[1]
        nu = param[0]
        lamb_value = nu * E/((1. - 2.*nu)*(1.+nu))
        mu_value = E/(2.*(1. + nu))
   
    lamb.assign(lamb_value)
    mu.assign(mu_value)
    
    angleGravity = param[2]
    f.assign( Constant( bodyForce(angleGravity) )) 

    K = assemble(a)
    rhs = assemble(L)
    
    bc.apply(K,rhs)
    
    u = Function(V)
    solve(K,u.vector(),rhs)
   
    
    if savevtk != None:
    # Save solution to file in VTK format
        File(savevtk) << u
    
    return u.vector().get_local()


def getAffineDecompositionElasticityBarFenics(V,Vbase):

    # Define boundary condition
    tol = 1E-14
    
    def clamped_boundary(x, on_boundary):
        return on_boundary and x[0] < tol
    
    bc = DirichletBC(V, Constant((0, 0)), clamped_boundary)
    
    # Define strain and stress
    
    def epsilon(u):
        return 0.5*(nabla_grad(u) + nabla_grad(u).T)
    
    def sigma1(u):
        return nabla_div(u)*Identity(d)
    
    def sigma2(u):
        return 2.0*epsilon(u)
    
    # Define variational problem
    u = TrialFunction(V)
    d = u.geometric_dimension()  # space dimension
    v = TestFunction(V)
    f1 = Constant((0 ,0))
    f2 = Constant((0 ,0))
    T = Constant((0, 0))
    a1 = inner(sigma1(u), epsilon(v))*dx
    a2 = inner(sigma2(u), epsilon(v))*dx
    L1 = dot(f1, v)*dx + dot(T, v)*ds
    L2 = dot(f2, v)*dx + dot(T, v)*ds
    
    
    f1.assign( Constant( bodyForce(0.0) ) ) # cosine part 
    f2.assign( Constant( bodyForce(0.5*np.pi) ) ) # sine part 
    
    
    K = [assemble(a1), assemble(a2)]
    rhs = [assemble(L1) , assemble(L2)]
    
    bc.apply(K[0],rhs[0])
    bc.apply(K[1],rhs[1])
    
    VbaseT = Vbase.transpose()
    affineDecomposition = {'ANq0':0, 'ANq1':0, 'fNq0':0, 'fNq1':0, 'Aq0':0, 'Aq1':0, 'fq0':0, 'fq1':0}
    affineDecomposition['ANq0'] = np.dot(np.dot(VbaseT,K[0].array()), Vbase)
    affineDecomposition['ANq1'] = np.dot(np.dot(VbaseT,K[1].array()), Vbase)
    affineDecomposition['fNq0'] = np.dot(VbaseT,rhs[0].get_local())
    affineDecomposition['fNq1'] = np.dot(VbaseT,rhs[1].get_local())
    
    affineDecomposition['Aq0'] = K[0].array()
    affineDecomposition['Aq1'] = K[1].array()
    affineDecomposition['fq0'] = rhs[0].get_local()
    affineDecomposition['fq1'] = rhs[0].get_local()
    
    return affineDecomposition

def computeRBapprox(param,affineDecomposition,Vbase):
    
    AN = np.zeros((N,N))
    for j in range(2):
        AN += param[j]*affineDecomposition['ANq' + str(j)]

    fN = np.zeros(N)
    theta_f = [np.cos(param[2]), np.sin(param[2])]
    for j in range(2):
        fN += theta_f[j]*affineDecomposition['fNq' + str(j)] 
    
    uN = np.linalg.solve(AN,fN)
    

    uR=np.dot(Vbase,uN).flatten() 

    
    return uR


folder = "rb_bar_3param_negativePoisson/"

nparam = 3
nsFEA = 1000
nsRB = 10000
nsTest = 300
nsTotal = nsFEA + nsRB + nsTest

seed = 6
np.random.seed(seed)

angleGravity_min = -0.05*np.pi
angleGravity_max = 0.05*np.pi

nu_min = -0.9
nu_max = 0.48

E_min = 10.0
E_max = 15.0

paramLimits = np.array([[nu_min,nu_max],[E_min,E_max],[angleGravity_min,angleGravity_max]])

param = np.array([ [ paramLimits[j,0] + np.random.uniform()*(paramLimits[j,1] - paramLimits[j,0])   for j in range(nparam)]   for i in range(nsTotal)])
param[:,0:2] = elut.convertParam2(param[:,0:2], elut.youngPoisson2lame_planeStress)

lamb_min = np.min(param[:nsFEA,0])
lamb_max = np.max(param[:nsFEA,0])

mu_min = np.min(param[:nsFEA,1])
mu_max = np.max(param[:nsFEA,1])

angleGravity_min = np.min(param[:nsFEA,2])
angleGravity_max = np.max(param[:nsFEA,2])

paramLimits = np.array([[lamb_min,lamb_max],[mu_min,mu_max],[angleGravity_min,angleGravity_max]])

paramFEA = param[0:nsFEA,:]
paramRB = param[nsFEA:nsFEA + nsRB,:]
paramTest = param[nsFEA + nsRB:,:]


isLame = True

fespace = getFEspace()

Nh = fespace.dim()

snapshotsFEA = np.zeros((Nh,nsFEA))
snapshotsTest = np.zeros((Nh,nsTest))

for i in range(nsFEA):    
    print("building snapshot FEA %d"%(i))
    snapshotsFEA[:,i] = solveElasticityBarFenics(fespace,paramFEA[i,:], isLame)

for i in range(nsTest):
    print("building snapshot Test %d"%(i))
    # snapshotsTest[:,i] = solveElasticityBarFenics(fespace,paramTest[i,:],angleGravity, isLame)
    snapshotsTest[:,i] = solveElasticityBarFenics(fespace,paramTest[i,:], isLame, folder + 'displacement_' + str(i) + '.pvd')


# ========  RB approximation ==============================================
tol = 1.0e-7

U, sigma, ZT = np.linalg.svd(snapshotsFEA, full_matrices=False )
print(sigma)

N = 0
sigma2_acc = 0.0
threshold = (1.0 - tol*tol)*np.sum(sigma*sigma)
while sigma2_acc < threshold and N<nsFEA:
    sigma2_acc += sigma[N]*sigma[N]
    N += 1  

print(N)
input()
Vbase = U[:,:N]

affineDecomposition = getAffineDecompositionElasticityBarFenics(fespace,Vbase)

snapshotsRB = np.zeros((Nh,nsRB)) 
for i in range(nsRB):
    print("building snapshots RB %d"%(i))
    snapshotsRB[:,i] = computeRBapprox(paramRB[i,:],affineDecomposition,Vbase) # theta is same param

# snapshotsRB_fea = np.zeros((Nh,nsRB)) 
# for i in range(nsRB):
#     print("building snapshots RB %d"%(i))
#     snapshotsRB_fea[:,i] = solveElasticityBarFenics(fespace,paramRB[i,:], isLame)

# errors = np.linalg.norm(snapshotsRB - snapshotsRB_fea,axis=0)
# print(errors)
# print(np.mean(errors))

# -------------------------------------------
# Saving

np.savetxt(folder + "snapshotsFEA.txt",snapshotsFEA)        
np.savetxt(folder + "snapshotsTest.txt",snapshotsTest)
np.savetxt(folder + "snapshotsRB.txt",snapshotsRB)
np.savetxt(folder + "paramFEA.txt",paramFEA)
np.savetxt(folder + "paramTest.txt",paramTest)
np.savetxt(folder + "paramRB.txt",paramRB)
np.savetxt(folder + "paramLimits.txt",paramLimits)
np.savetxt(folder + "U.txt",U)
# np.savetxt(folder + "nodes.txt", fespace.mesh().coordinates()) # the mesh coordinates doesn't correspond to the nodes that stores dofs  
np.savetxt(folder + "nodes.txt", fespace.tabulate_dof_coordinates()[0::2,:] )  # we need to jump because there are repeated coordinates
np.savetxt(folder + "paramTest_poisson.txt",elut.convertParam2(paramTest[:,0:2], elut.composition(elut.lame2youngPoisson, elut.lameStar2lame)) )

for j in range(2):
    label = "ANq" + str(j)
    np.savetxt(folder + label + ".txt", affineDecomposition[label])
    
    label = "Aq" + str(j)        
    np.savetxt(folder + label + ".txt", affineDecomposition[label])

for j in range(2):
    label = "fNq" + str(j)
    np.savetxt(folder + label + ".txt", affineDecomposition[label])
    
    label = "fq" + str(j)
    np.savetxt(folder + label + ".txt", affineDecomposition[label])

# ==============================================================================

# tolList = [0.0, 1.e-8,1.e-6,1.e-4]

# for tol in tolList:
#     len(sigma)
#     if(tol<1.e-26):
#         N = ns
#     else:
#         sigma2_acc = 0.0
#         threshold = (1.0 - tol*tol)*np.sum(sigma*sigma)
#         N = 0
#         while sigma2_acc < threshold and N<ns:
#             sigma2_acc += sigma[N]*sigma[N]
#             N += 1  
  
#     V = U[:,:N]
    
#     affineDecomposition = getAffineDecompositionElasticityBarFenics(fespace,V)
    
#     tolEff = np.sqrt(1.0 - np.sum(sigma[0:N]*sigma[0:N])/np.sum(sigma*sigma))
    
#     errors = np.zeros(nsTest)
#     for i in range(nsTest):
#         nu =  paramTest[i,0]
#         E = paramTest[i,1] 
#         lamb = nu * E/((1. - 2.*nu)*(1.+nu))
#         mu = E/(2.*(1. + nu))
        
#         uR = computeRBapprox([lamb,mu],affineDecomposition)
                    
#         u = solveElasticityBarFenics(fespace,paramTest[i,:]) 
        
#         errors[i] = np.linalg.norm(uR - u)

#     print(" ====== Summary test RB for randomly sampled tested parameters (ns=%d,N=%d,eps=%d,epsEff) ==== ",ns,N,tol,tolEff)
#     print("errors = ", errors)
#     print("avg error = ", np.average(errors))
#     print("================================================================\n") 

