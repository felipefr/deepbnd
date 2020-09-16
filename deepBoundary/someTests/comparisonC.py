import sys, os
import numpy as np
import matplotlib.pyplot as plt

folder = "../data1/"


## First conclusion is that the integration with 1 quadrature point is sufficient, since we both function are in P1 spaces

# nq = 6
# C = []
# sig = []
# U = []

# for k in range(nq):
#     C.append(np.loadtxt('C' + str(k+1) + '.txt'))
#     sig_temp, U_temp = np.linalg.eig(C[k])    
#     asort = np.argsort(sig_temp)
#     sig.append(sig_temp[asort[::-1]])
#     U.append(U_temp[:,asort[::-1]])


# dist = np.zeros((nq,nq))

# for i in range(nq):
#     for j in range(i+1,nq):
#         dist[i,j] = np.linalg.norm(C[i]-C[j],'fro')
#         dist[i,j] = np.linalg.norm(C[i]-C[j],np.inf)
        


# plt.figure(1)
# plt.imshow(dist)
# plt.colorbar()
# plt.savefig("distC.png")


C1 = np.loadtxt('C1.txt')
C2 = np.loadtxt('C_refine_1.txt')
C3 = np.loadtxt('C_refine_mult_1.txt')

C = [C1,C2,C3]

sig = []
U = []

for k in range(3):
    sig_temp, U_temp = np.linalg.eig(C[k])    
    asort = np.argsort(sig_temp)
    sig.append(sig_temp[asort[::-1]])
    U.append(U_temp[:,asort[::-1]])


print(np.linalg.norm(C1 - C2))

plt.figure(1,(8,15))
plt.subplot('311')
plt.title("C - Cref2")
plt.imshow(C1-C3)
plt.colorbar()
plt.subplot('312')
plt.title("Cref - Cref2")
plt.imshow(C2-C3)
plt.colorbar()
plt.subplot('313')
plt.title("C - Cref")
plt.imshow(C1-C2)
plt.colorbar()
plt.savefig("distC_all.png")


plt.figure(2)
plt.title("Spectrum")
plt.plot(sig[0], '-', label = 'C')
plt.plot(sig[1], '-', label = 'Cref')
plt.plot(sig[2], '-', label = 'Cref2')
plt.yscale('log')
plt.xlabel('eigenvalue index')
plt.ylabel('eigenvalue error rel')
plt.grid()
plt.legend()
plt.savefig("spectrum_log_all.png")

plt.figure(3)
plt.title("Error abs : Cref - Cref2")
plt.plot(np.abs(sig[0] - sig[2]), 'o', label = 'C_x_Cref2')
plt.plot(np.abs(sig[1] - sig[2]), 'o', label = 'Cref_x_Cref2')
plt.yscale('log')
plt.xlabel('eigenvalue index')
plt.ylabel('eigenvalue error')
plt.grid()
plt.legend()
plt.savefig("error_abs_spectrum_log_all.png")


plt.figure(4)
plt.title("Error rel : Cref - Cref2")
plt.plot(np.abs(sig[0] - sig[2])/np.abs(sig[2]), 'o', label = 'C_x_Cref2')
plt.plot(np.abs(sig[1] - sig[2])/np.abs(sig[2]), 'o', label = 'Cref_x_Cref2')
plt.yscale('log')
plt.xlabel('eigenvalue index')
plt.ylabel('eigenvalue error rel')
plt.grid()
# plt.legend()
plt.savefig("error_rel_spectrum_log_all.png")

plt.figure(5)
plt.plot(sig[2],sig[0], 'x', label = 'C')
plt.plot(sig[2],sig[1], 'x', label = 'Cref')
plt.plot([1e-8,1],[1e-8,1], 'k-')
plt.xlabel('eigenvalue Cref2')
plt.ylabel('eigenvalue (C or Cref)')
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.legend()
plt.savefig("straightline_spectrum_all.png")

ns = 400
cos1 = np.zeros(ns)
cos2 = np.zeros(ns)

getCos = lambda a,b : np.abs(np.dot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b))

for i in range(ns):
    cos1_ = np.zeros(5)
    cos2_ = np.zeros(5)
    
    # if i>1:
    #     cos1_[0] = getCos(U[2][:,i],U[0][:,i-2])
    #     cos2_[0] = getCos(U[2][:,i],U[1][:,i-2])
    
    # if i>0:
    #     cos1_[1] = getCos(U[2][:,i],U[0][:,i-1])
    #     cos2_[1] = getCos(U[2][:,i],U[1][:,i-1])
    
    # if i<ns-1:
    #     cos1_[3] = getCos(U[2][:,i],U[0][:,i+1])
    #     cos2_[3] = getCos(U[2][:,i],U[1][:,i+1])

    # if i<ns-2:
    #     cos1_[4] = getCos(U[2][:,i],U[0][:,i+2])
    #     cos2_[4] = getCos(U[2][:,i],U[1][:,i+2])
        
        
    cos1_[2] = getCos(U[2][:,i],U[0][:,i])
    cos2_[2] = getCos(U[2][:,i],U[1][:,i])
    
    cos1[i] = np.max(cos1_)
    cos2[i] = np.max(cos2_)
    
plt.figure(6)
plt.plot(cos1 , label = 'C_x_Cref2')
plt.plot(cos2 , label = 'Cref_x_Cref2')
plt.xlabel('index eigenvalue')
# plt.ylabel('cosine eigenvectors (window max 2)')
plt.ylabel('cosine eigenvectors')
# plt.xscale('')
# plt.yscale('log')
plt.grid()
plt.legend()
plt.savefig("cos_U_all.png")


plt.show()
