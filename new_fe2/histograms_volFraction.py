import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0,'../utils/')

# import generation_deepBoundary_lib as gdb
import matplotlib.pyplot as plt
import numpy as np

import myHDF5 as myhd
from scipy.stats import norm

# Test Loading 
Ny = 24


fac = 4
Ly = 0.5
Lx = fac*Ly
Nx = fac*Ny
H = Lx/Nx # same as Ly/Ny
x0 = y0 = 0.0
lcar = (1/9)*H # more less the same order than the RVE
r0 = 0.2*H
r1 = 0.4*H
Vfrac = 0.282743 # imporsed Vfrac
rm = H*np.sqrt(Vfrac/np.pi)

rootDataPath = open('../../rootDataPath.txt','r').readline()[:-1]
print(rootDataPath)

folder = rootDataPath + '/fe2/big_DNS/DNS_{0}/'.format(Ny)


ellipseData = myhd.loadhd5(folder + 'ellipseData_RVEs.hd5', 'ellipseData')


volFracs = np.zeros(len(ellipseData)) 
for i in range(len(ellipseData)):
    r = ellipseData[i,:,2]
    volFracs[i] = np.pi*np.sum(r*r)/6**2
    
# Fit a normal distribution to the data:
xmin, xmax = volFracs.min() , volFracs.max() 
x = np.linspace(xmin, xmax, 100)

mu, std = norm.fit(volFracs)    
p = norm.pdf(x, mu, std)

plt.figure(1)
plt.title('Histogram for DNS Ny %d'%(Ny))
plt.hist(volFracs, bins=25, density=True, alpha=0.6, color='g', label = 'histogram')
plt.plot(x, p, 'k', linewidth=2, label = 'N(%.6f, %.6f)'%(mu, std))
plt.plot(2*[Vfrac] , [0, 38], '--', label = 'Vf = %.6f'%(Vfrac) )
plt.xlabel('Volume Fraction')
plt.ylabel('Probability Density')
plt.legend()
plt.savefig('histogramsVolFracs_ny{0}.png'.format(Ny))
plt.show()


# suptitle = 'Error_rel_ny24_vs_fullPer{0}'.format(jump) 

# plt.figure(2)
# plt.title(suptitle)
# for i in range(n-1):
#     plt.plot(errors[i,1::2], '-o', label = tangent_labels[i+1]) # jump tangent_label 0 (reference)
# plt.yscale('log')
# plt.xticks([0,1,2,3,4,5,6],labels = norms_label)
# plt.xlabel('Relative Norms')
# plt.grid()
# plt.legend(loc = 'best')
# plt.savefig(folder + suptitle + '.png')

