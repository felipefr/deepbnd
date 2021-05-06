import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0,'../../utils/')

# import generation_deepBoundary_lib as gdb
import matplotlib.pyplot as plt
import numpy as np

# import myTensorflow as mytf
from timeit import default_timer as timer

import h5py
import myHDF5 as myhd
import matplotlib.pyplot as plt
import meshUtils as meut
from dolfin import *

# Test Loading 

rootDataPath = open('../../../rootDataPath.txt','r').readline()[:-1]
print(rootDataPath)

Ny_DNS = 72

folder = rootDataPath + '/new_fe2/DNS/DNS_%d_old/'%Ny_DNS

tangent_labels = ['DNS', 'dnn_big', 'reduced_per', 'full_per']
solutions = [ 'barMacro_DNS.xdmf', 
             'multiscale/barMacro_Multiscale_dnn_big.xdmf', 
             'multiscale/barMacro_Multiscale_reduced_per.xdmf',
              'multiscale/barMacro_Multiscale_full.xdmf'] # temporary
         
solutions = [folder + f for f in solutions]

meshDNSfile =  folder + 'mesh.xdmf'
meshMultiscaleFile = folder + 'multiscale/meshBarMacro_Multiscale.xdmf'

meshMultiscale = Mesh()
with XDMFFile(meshMultiscaleFile) as infile:
    infile.read(meshMultiscale)
        

Uh_mult = VectorFunctionSpace(meshMultiscale, "CG", 2)

uhs = []
for f in solutions:    
    uh = Function(Uh_mult)
    print(f)
    with XDMFFile(f) as infile:
        infile.read_checkpoint(uh, 'u', 0)
        uhs.append(uh)

# uhs[0] = interpolate(uhs[0],Uh_mult)

pA = Point(2.0,0.0)
pB = Point(2.0,0.5)


norms_label = ['L2', '||u(A)||', '||u(B)||', '|u_1(A)|', '|u_1(B)|', '|u_2(A)|', '|u_2(B)|']
norms = [lambda x: norm(x), lambda x: np.linalg.norm(x(pA)), lambda x: np.linalg.norm(x(pB)), 
          lambda x: abs(x(pA)[0]), lambda x: abs(x(pB)[0]), lambda x: abs(x(pA)[1]), lambda x: abs(x(pB)[1])]
norms_ref = np.array([N(uhs[0]) for N in norms]) 

e = Function(Uh_mult)
n = len(solutions) - 1
errors = np.zeros((n,len(norms)))
errors_rel = np.zeros((n,len(norms)))

for i in range(n):
    e.vector().set_local(uhs[i+1].vector().get_local()[:]-uhs[0].vector().get_local()[:])
    for j, N in enumerate(norms):
        errors[i,j] = N(e)
    
for i in range(n):
    errors_rel[i,:] = errors[i,:]/norms_ref[:]
        


suptitle = 'Error_DNS_%d_bar_vs_DNS'%Ny_DNS 

np.savetxt(folder + 'errors_{0}.txt'.format(suptitle), errors)

plt.figure(1)
plt.title(suptitle)
for i in range(n):
    plt.plot(errors[i,:], '-o', label = tangent_labels[i+1]) # jump tangent_label 0 (reference)
plt.yscale('log')
plt.xticks([0,1,2,3,4,5,6],labels = norms_label)
plt.xlabel('Norms')
plt.grid()
plt.legend(loc = 'best')
plt.savefig(folder + suptitle + '.png')

suptitle = 'Error_DNS_%d_rel_vs_DNS'%Ny_DNS 

plt.figure(2)
plt.title(suptitle)
for i in range(n):
    plt.plot(errors[i,:], '-o', label = tangent_labels[i+1]) # jump tangent_label 0 (reference)
plt.yscale('log')
plt.xticks([0,1,2,3,4,5,6],labels = norms_label)
plt.xlabel('Relative Norms')
plt.grid()
plt.legend(loc = 'best')
plt.savefig(folder + suptitle + '.png')


