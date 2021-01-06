import os, sys
sys.path.insert(0,'../../utils/')

import matplotlib.pyplot as plt
import numpy as np
from dolfin import *
import myHDF5 as myhd
import meshUtils as meut
import ioFenicsWrappers as iofe
# import plotUtils as plut

dotProduct = lambda u,v, ds : assemble(inner(u,v)*ds)

nameMeshRefBnd = 'boundaryMesh_2.xdmf'

meshRef = meut.degeneratedBoundaryRectangleMesh(x0 = -1.0, y0 = -1.0, Lx = 2.0 , Ly = 2.0, Nb = 30)
meshRef.generate()
meshRef.write(nameMeshRefBnd, 'fenics')
Mref = meut.EnrichedMesh(nameMeshRefBnd)
Vref = VectorFunctionSpace(Mref,"CG", 1)
# usol = Function(Vref)

# Mref = meut.EnrichedMesh(nameMeshRefBnd)
# Vref = VectorFunctionSpace(Mref,"CG", 1)
dsRef = Measure('ds', Mref) 

C = {}
Wbasis = {}
eig = {}
basis = {}
for sampling in ['veryRegular','mixedSampled','fullSampled']:
    folder = ["/Users", "/home"][1] + "/felipefr/EPFL/newDLPDEs/DATA/deepBoundary/RBsensibility/{0}/shear/".format(sampling)
    nameC = folder + 'C.h5'
    nameWbasis = folder + 'Wbasis.h5'
    basis[sampling] = Function(Vref)
    
    C[sampling] = myhd.loadhd5(nameC,'C')
    eig[sampling] = np.linalg.eigh([C[sampling]])[0].reshape(1000)
    asort = np.argsort(eig[sampling])
    eig[sampling] =   eig[sampling][asort[::-1]]
    Wbasis[sampling] = myhd.loadhd5(nameWbasis,'Wbasis')

Nmax = 240

error = np.zeros((3,Nmax))
angle = np.zeros((3,Nmax))
 
for i in range(Nmax):
    for j,s1 in enumerate(['veryRegular','mixedSampled','fullSampled']):
        basis[s1].vector().set_local(Wbasis[s1][i,:])
        for k,s2 in enumerate(['veryRegular','mixedSampled','fullSampled']):
            if(k>j):
                error[j+k-1 , i] = dotProduct(basis[s1] - basis[s2],basis[s1] - basis[s2], dsRef)
                angle[j+k-1 , i] = dotProduct(basis[s1],basis[s2], dsRef)

for s1 in ['veryRegular','mixedSampled','fullSampled']:
    basisList = []
    for i in range(10):
        b = Function(Vref)
        b.vector().set_local(Wbasis[s1][i,:])
        basisList.append(b)
    
    iofe.postProcessing_temporal(basisList, 'basis_{0}.xdmf'.format(s1) , comm = MPI.comm_world)


fig = plt.figure(1, (6,4))
plt.title('Spectrum (Periodic - Shear)')
plt.xlabel('N')
plt.ylabel(r'eigenvalues')
plt.grid()
for s in ['veryRegular','mixedSampled','fullSampled']:
    plt.plot(eig[s][:15] , label = s)
    
plt.legend(loc = 'best')
# plt.ylim(1.0e-5,1.0e-1)
plt.yscale('log')
plt.savefig("spectrum_periodic_shear_15.pdf")


fig = plt.figure(2, (6,4))
plt.title('Error Basis')
plt.xlabel('N')
plt.ylabel(r'norm L2(bnd)')
# plt.ylim([1.0e-11,0.1])
# plt.yscale('log')
plt.grid()
for i in range(3):
    label = ['Mixed Sampled - Regular', 'Full Sampled - Regular', 'Mixed - Full Sampled'][i]
    plt.plot(error[i,:20], label = label) 
plt.legend(loc=8)
plt.savefig("errorBasis_20.pdf")


fig = plt.figure(3, (6,4))
plt.title('Angle Basis')
plt.xlabel('N')
plt.ylabel(r'norm L2(bnd)')
# plt.ylim([1.0e-11,0.1])
# plt.yscale('log')
plt.grid()
for i in range(3):
    label = ['Mixed Sampled - Regular', 'Full Sampled - Regular', 'Mixed - Full Sampled'][i]
    plt.plot(angle[i,:], label = label) 
plt.legend(loc=8)
plt.savefig("angleBasis_all.pdf")

plt.show()
    
    