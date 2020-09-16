import numpy as np
import matplotlib.pyplot as plt
import os

n = 76
Nbasis = np.arange(10,n)
sigma_ref = np.loadtxt('sigma_ref.txt')
sigmaList = np.zeros((len(Nbasis),4))
uBoundList = np.zeros((len(Nbasis),160))
normList = np.zeros(len(Nbasis))
u_ref = np.loadtxt('uBound_ref.txt')

for i, N in enumerate(Nbasis):
    sigmaList[i,:] = np.loadtxt('sigma_' + str(N) + '.txt')
    uBoundList[i,:] = np.loadtxt('uBound_' + str(N) + '.txt')
    normList[i] = np.linalg.norm(uBoundList[i,:] - u_ref)
    
plt.figure(1)
plt.plot(Nbasis, sigmaList[:,0] - sigma_ref[0], '-o')
plt.grid()

plt.figure(2)
plt.plot(Nbasis, normList , '-o')
plt.yscale('log')
plt.grid()

# plt.figure(1,(8,10))
# plt.suptitle("Sigma Convergence")
# for i in range(3):
#     j = [0,3,1][i]
#     plt.subplot('31' + str(i+1))
#     plt.plot(sigma[:,j],'o-', label = 'M')
#     # plt.plot(sigma_linear[:,j],'o-', label = 'linear')
#     plt.plot(sigma_linear_zero[:,j],'o-', label = 'L')
#     plt.plot(sigma_periodic[:,j],'o-', label = 'P')
#     plt.xlabel('n-1')
#     plt.ylabel('sigma_' + str(j))
#     plt.legend(loc = 'best')
#     plt.grid()

# plt.savefig('SigmaConvergence.png')
    
# plt.figure(2,(8,8))
# plt.suptitle("Principal Stress Convergence")
# for i in range(2):
#     plt.subplot('21' + str(i+1))
#     plt.plot(lamb[:,i],'o-', label = 'M')
#     # plt.plot(lamb_linear[:,i],'o-', label = 'linear')
#     plt.xlabel('n-1')
#     plt.plot(lamb_linear_zero[:,i],'o-', label = 'L')
#     plt.plot(lamb_periodic[:,i],'o-', label = 'P')
#     plt.ylabel('lamb_' + str(i))
#     plt.legend(loc = 'best')
#     plt.grid()

# plt.savefig('PrincipalStressConvergence.png')

    
# plt.figure(3,(8,8))
# plt.suptitle("Homogenised Tangent")
# plt.subplot('211' )
# plt.plot(chom[:,9],'o-', label = 'M')
# plt.plot(chom_linear[:,9],'o-', label = 'L')
# plt.plot(chom_periodic[:,9],'o-', label = 'P')
# plt.ylabel('Ehom')
# plt.xlabel('n-1')
# plt.legend(loc = 'best')
# plt.grid()

# plt.subplot('212' )
# plt.plot(chom[:,10],'o-', label = 'M')
# plt.plot(chom_linear[:,10],'o-', label = 'L')
# plt.plot(chom_periodic[:,10],'o-', label = 'P')
# plt.ylabel('nu_hom')
# plt.xlabel('n-1')
# plt.legend(loc = 'best')
# plt.grid()

# plt.savefig('homogenisedTangent.png')

# plt.show()