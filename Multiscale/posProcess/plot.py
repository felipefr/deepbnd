import numpy as np
import matplotlib.pyplot as plt

sigma = np.loadtxt("sigma_2.txt")
sigmaL = np.loadtxt("sigmaL_2.txt")

plt.figure(1,(8,10))
for i in range(3):
    j = [0,3,1][i]
    plt.subplot('31' + str(i+1))
    mean = np.mean(sigma[:,j])*np.ones(len(sigma))
    meanL = np.mean(sigmaL[:,j])*np.ones(len(sigma))
    
    plt.plot(sigma[:,j],'o', label = 'whole')
    plt.plot(sigmaL[:,j],'o', label = 'L')
    plt.plot(mean,'-', label = 'mean whole')
    plt.plot(meanL,'-', label = 'mean L')
    plt.ylabel('sigma_' + str(j))
    plt.legend(loc = 'best')
    plt.grid()
    

plt.show()