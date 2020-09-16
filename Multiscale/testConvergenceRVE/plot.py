import numpy as np
import matplotlib.pyplot as plt
import os

def principalStresses(sigma):
    n = sigma.shape[0]
    lamb = np.zeros((n,2))
    
    for i in range(n):
        lamb[i,:] = np.linalg.eig(sigma[i,:].reshape((2,2)))[0]

    return lamb

folder = "./simuls_10/"
radFile = folder + "test_{0}.{1}"
sigmaFile = radFile.format('sigma','txt')

replications = 9
if replications > 0:
    os.system('rm ' + sigmaFile)
    for i in range(1,replications+1):
        sigmaFile_i = radFile.format('sigma_' + str(i),'txt')
        os.system('cat ' + sigmaFile_i + ' >> ' + sigmaFile)
        # os.system('rm ' + sigmaFile_i)

sigma = np.loadtxt(sigmaFile)
lamb = principalStresses(sigma)

sigmaFile = radFile.format('sigma_linear','txt')

if replications > 0:
    os.system('rm ' + sigmaFile)
    for i in range(1,replications+1):
        sigmaFile_i = radFile.format('sigma_linear_' + str(i),'txt')
        os.system('cat ' + sigmaFile_i + ' >> ' + sigmaFile)
        # os.system('rm ' + sigmaFile_i)
    
    
sigma_linear = np.loadtxt(sigmaFile)
lamb_linear = principalStresses(sigma_linear)


sigmaFile = radFile.format('sigma_linear_zero_mean','txt')
if replications > 0:
    os.system('rm ' + sigmaFile)
    for i in range(1,replications+1):
        sigmaFile_i = radFile.format('sigma_linear_zero_mean_' + str(i),'txt')
        os.system('cat ' + sigmaFile_i + ' >> ' + sigmaFile)
        # os.system('rm ' + sigmaFile_i)
    
    
sigma_linear_zero = np.loadtxt(sigmaFile)
lamb_linear_zero = principalStresses(sigma_linear_zero)



chomFile = radFile.format('Chom_linear','txt')
if replications > 0:
    os.system('rm ' + chomFile)
    for i in range(1,replications+1):
        chomFile_i = radFile.format('Chom_linear_' + str(i),'txt')
        os.system('cat ' + chomFile_i + ' >> ' + chomFile)
        # os.system('rm ' + sigmaFile_i)
    
    
chom_linear = np.loadtxt(chomFile)

chomFile = radFile.format('Chom','txt')
if replications > 0:
    os.system('rm ' + chomFile)
    for i in range(1,replications+1):
        chomFile_i = radFile.format('Chom_' + str(i),'txt')
        os.system('cat ' + chomFile_i + ' >> ' + chomFile)
        # os.system('rm ' + sigmaFile_i)
    
    
chom = np.loadtxt(chomFile)

## files periodic
sigmaFile = radFile.format('sigma_periodic','txt')
if replications > 0:
    os.system('rm ' + sigmaFile)
    for i in range(1,replications+1):
        sigmaFile_i = radFile.format('sigma_periodic_' + str(i),'txt')
        os.system('cat ' + sigmaFile_i + ' >> ' + sigmaFile)
        # os.system('rm ' + sigmaFile_i)

sigma_periodic = np.loadtxt(sigmaFile)
lamb_periodic = principalStresses(sigma_periodic)

chomFile = radFile.format('Chom_periodic','txt')
if replications > 0:
    os.system('rm ' + chomFile)
    for i in range(1,replications+1):
        chomFile_i = radFile.format('Chom_periodic_' + str(i),'txt')
        os.system('cat ' + chomFile_i + ' >> ' + chomFile)
        # os.system('rm ' + sigmaFile_i)

chom_periodic = np.loadtxt(chomFile)

plt.figure(1,(12,10))
plt.title("Sigma Convergence")

fac = 0.5*(sigma[-1,0] + sigma_linear_zero[-1,0])/sigma_11_N[-1,-1]

for i in range(1):
    j = [0,3,1][i]
    # plt.subplot('31' + str(i+1))
    plt.plot(sigma[:,j],'o-', label = 'M')
    # plt.plot(sigma_linear[:,j],'o-', label = 'linear')
    plt.plot(sigma_linear_zero[:,j],'o-', label = 'L')
    plt.plot(sigma_periodic[:,j],'o-', label = 'P')
    plt.xlabel('n-1')
    plt.ylabel('sigma_' + str(j))
    plt.legend(loc = 'best')
    plt.grid()

plt.plot(Nbasis*8/Nbasis[-1], fac*np.mean(sigma_11_N, axis = 0),'-ro', label = 'stress UN')
plt.plot(Nbasis*8/Nbasis[-1], fac*np.mean(sigma_11_N, axis = 0) + np.std(sigma_11_N, axis = 0), 'r--')
plt.plot(Nbasis*8/Nbasis[-1], fac*np.mean(sigma_11_N, axis = 0) - np.std(sigma_11_N, axis = 0),'r--')
plt.plot(Nbasis*8/Nbasis[-1], fac*np.mean(sigma_11_R, axis = 0),'-bo', label = 'stress UR')
plt.plot(Nbasis*8/Nbasis[-1], fac*np.mean(sigma_11_R, axis = 0) + np.std(sigma_11_R, axis = 0), 'b--')
plt.plot(Nbasis*8/Nbasis[-1], fac*np.mean(sigma_11_R, axis = 0) - np.std(sigma_11_R, axis = 0),'b--')

plt.savefig('SigmaConvergence.png')
    
plt.figure(2,(8,8))
plt.suptitle("Principal Stress Convergence")
for i in range(2):
    plt.subplot('21' + str(i+1))
    plt.plot(lamb[:,i],'o-', label = 'M')
    # plt.plot(lamb_linear[:,i],'o-', label = 'linear')
    plt.xlabel('n-1')
    plt.plot(lamb_linear_zero[:,i],'o-', label = 'L')
    plt.plot(lamb_periodic[:,i],'o-', label = 'P')
    plt.ylabel('lamb_' + str(i))
    plt.legend(loc = 'best')
    plt.grid()

plt.savefig('PrincipalStressConvergence.png')

    
plt.figure(3,(8,8))
plt.suptitle("Homogenised Tangent")
plt.subplot('211' )
plt.plot(chom[:,9],'o-', label = 'M')
plt.plot(chom_linear[:,9],'o-', label = 'L')
plt.plot(chom_periodic[:,9],'o-', label = 'P')
plt.ylabel('Ehom')
plt.xlabel('n-1')
plt.legend(loc = 'best')
plt.grid()

plt.subplot('212' )
plt.plot(chom[:,10],'o-', label = 'M')
plt.plot(chom_linear[:,10],'o-', label = 'L')
plt.plot(chom_periodic[:,10],'o-', label = 'P')
plt.ylabel('nu_hom')
plt.xlabel('n-1')
plt.legend(loc = 'best')
plt.grid()

plt.savefig('homogenisedTangent.png')

plt.show()