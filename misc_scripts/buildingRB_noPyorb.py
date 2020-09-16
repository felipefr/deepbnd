#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:28:24 2019

@author: felipefr
"""

# First try of construction of the RB basis without using pyorb, relying just in the feamat framework, whose wrappers are 
# defined in place. This is for the problem of elasticity. 
# Consistent results, i.e., full basis leads to zero error for the same set of parameters used to build the simulation snapshots. 
# The error decreases as expected by enlarging the basis, and so on. 

# Todo: Try to become independent of matlab by writing a simple wrapper from octave. 
# Also, the problem is simple enough to be solved by using one of my finite element solvers in pure python. 


import matlab.engine
import numpy as np

matlab_library_path = '../../feamat/'
Meng = matlab.engine.start_matlab()
Meng.addpath( Meng.genpath(matlab_library_path))

def computeRBapprox(theta,affineDecomposition):
    
    AN = np.zeros((N,N))
    for j in range(2):
        AN += theta[j]*affineDecomposition['ANq' + str(j+1)]

    fN = np.zeros(N)
    for j in range(1):
        fN += np.array(affineDecomposition['fNq' + str(j+1)]).flatten() 
    

    uN = np.linalg.solve(AN,fN)
    
    uR = np.dot(V,uN).flatten()
    
    return uR



meshName =  'bar2.msh'
nu_min = 0.1 
nu_max = 0.40
E_min = 100.0
E_max = 150.0

paramLimits = np.array([[nu_min,nu_max],[E_min,E_max]])
nParam = len(paramLimits)

simulData = Meng.setPreliminariesElasticityBar(meshName)


Nh = 2*len(simulData['fespace']['nodes'])
ns = 500

nsTest = 50
seed = 3
np.random.seed(seed)

#snapshots = np.zeros((Nh,ns))


#param = np.array([ [ paramLimits[j,0] + np.random.uniform()*(paramLimits[j,1] - paramLimits[j,0])   for j in range(nParam)]   for i in range(ns)])

#for i in range(ns):    
#    print("building snapshot %d"%(i))
#    snapshots[:,i] = np.array(Meng.solveElasticityBar(matlab.double(param[i,:].tolist()),simulData)).flatten()
#    out = Meng.solveElasticityBar(matlab.double(param[i,:].tolist()),meshName)
#    snapshots[:,i] = np.array(out['u']).flatten()
#    print(out['fespace'])
#    print(out['dirich'])
    

    
snapshots=  np.loadtxt("snapshots.txt")
param=  np.loadtxt("param.txt")
#snapshots = np.concatenate((snapshots,snapshots_loaded),axis = 1)
#ns = len(snapshots[0])

#np.savetxt("snapshots.txt",snapshots)
#np.savetxt("param.txt",param)

U, sigma, ZT = np.linalg.svd(snapshots, full_matrices=False )

np.savetxt("U.txt",U)
np.savetxt("sigma.txt",param)


# ========  Test RB for all snapshots ==============================================
N = 20
V = U[:,:N]
affineDecomposition = Meng.getAffineDecomposition_elasticityBar(matlab.double(V.tolist()),simulData);
snapshotsRecovered = np.zeros((Nh,ns))

for j in range(2):
    label = "ANq" + str(j)
    np.savetxt(label + ".txt", affineDecomposition[label])
    
for j in range(1):
    label = "fNq" + str(j)
    np.savetxt(label + ".txt", affineDecomposition[label])

theta = np.zeros((ns,2))

for i in range(ns):
    print('recovering snapshot ' + str(i))
    nu = param[i,0]
    E = param[i,1]
    lamb = nu * E/((1. - 2.*nu)*(1.+nu))
    mu = E/(2.*(1. + nu))
    
    theta[i,0] = 2.0*mu
    theta[i,1] = lamb
  
    snapshotsRecovered[:,i] = computeRBapprox([2.*mu,lamb],affineDecomposition)

np.savetxt("snapshotsRecovered.txt",snapshotsRecovered)

print("Summary test RB for all snapshots")
errors = np.linalg.norm(snapshots - snapshotsRecovered,axis = 0)
print("errors = ", errors)
print("avg error = ", np.average(errors))
#
# -------------------------------------------

#snapshotsTest = np.zeros((Nh,nsTest))
#paramTest = np.array([ [ paramLimits[j,0] + np.random.uniform()*(paramLimits[j,1] - paramLimits[j,0])   for j in range(nParam)]   for i in range(nsTest)])
#
#for i in range(nsTest):
#    print("building snapshot test %d"%(i))
#    snapshotsTest[:,i] = np.array(Meng.solveElasticityBar(matlab.double(paramTest[i,:].tolist()),simulData)).flatten()
#        
#np.savetxt("snapshotsTest.txt",snapshotsTest)
#np.savetxt("paramTest.txt",paramTest)

# ==============================================================================

#tolList = [0.0, 1.e-8,1.e-6,1.e-4]
#
#for tol in tolList:
#    len(sigma)
#    if(tol<1.e-26):
#        N = ns
#    else:
#        sigma2_acc = 0.0
#        threshold = (1.0 - tol*tol)*np.sum(sigma*sigma)
#        N = 0
#        while sigma2_acc < threshold and N<ns:
#            sigma2_acc += sigma[N]*sigma[N]
#            N += 1  
#   
#    V = U[:,:N]
#    
#    affineDecomposition = Meng.getAffineDecomposition_elasticityBar(matlab.double(V.tolist()),simulData);
#    
#    tolEff = np.sqrt(1.0 - np.sum(sigma[0:N]*sigma[0:N])/np.sum(sigma*sigma))
#    
#    errors = np.zeros(nsTest)
#    for i in range(nsTest):
#        nu, E = [ paramLimits[j,0] + np.random.uniform()*(paramLimits[j,1] - paramLimits[j,0]) for j in range(nParam)] 
#        lamb = nu * E/((1. - 2.*nu)*(1.+nu))
#        mu = E/(2.*(1. + nu))
#        
#        AN = 2.0*mu*affineDecomposition['ANq0'] + lamb*affineDecomposition['ANq1'] 
#        fN = affineDecomposition['fNq0'] 
#        
#        uN = np.linalg.solve(AN,fN)
#        
#        uR = np.dot(V,uN).flatten()
#        
#        u = np.array(Meng.solveElasticityBar(matlab.double([nu,E]),simulData)).flatten()
#        
#        errors[i] = np.linalg.norm(uR - u)
#
#    print(" ====== Summary test RB for randomly sampled tested parameters (ns=%d,N=%d,eps=%d,epsEff) ==== ",ns,N,tol,tolEff)
#    print("errors = ", errors)
#    print("avg error = ", np.average(errors))
#    print("================================================================\n") 

Meng.quit()

