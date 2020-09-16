from __future__ import print_function
import numpy as np
from fenics import *
from dolfin import *
from ufl import nabla_div
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, '../../utils/')
import copy

import elasticity_utils as elut

import fenicsWrapperElasticity as fela 
import ioFenicsWrappers as iofe

from timeit import default_timer as timer
import myCoeffClass as coef

# metadata={"quadrature_degree": 1} 

def getMapParamToDomain(paramAux,p):
    paramAux[0:9,0] = p[0] ; paramAux[0:9,1] = p[2]
    paramAux[9,0] = p[1]; paramAux[9,1] = p[3]

zeroParam = 4*[0.0]

# Mesh construction
folder = 'saves_twoBCs/'
MeshFileRad = 'mesh'
MeshFileExt = 'xdmf'
referenceGeo = 'reference.geo'
labelsPhysicalGroups = {'line' : 'faces', 'triangle' : 'regions'}

# defaultMeshParam = iofe.getDefaultParameters_given_hRef(hRef = 0.125)
# meshFile = folder + 'mesh_veryCoarse.xdmf'
# iofe.exportMeshXDMF_fromReferenceGeo(defaultMeshParam, referenceGeo = 'reference_twoBCs.geo', meshFile = meshFile)

# os.system("mv mesh.msh mesh0.msh")
# os.system("mv mesh0.msh meshTemp.msh")

Nrefine = 7
# for i in range(Nrefine):
#     print("refining {0}th times".format(i+1))
#     os.system("gmsh meshTemp.msh -refine -format 'msh2' -o meshTemp2.msh")       
#     iofe.exportMeshXDMF_fromGMSH('meshTemp2.msh', meshFile = '{0}{1}_{2}.{3}'.format(folder, MeshFileRad, i + 1, MeshFileExt), labels = labelsPhysicalGroups)
#     os.system("mv meshTemp2.msh meshTemp.msh")
   
nu0 = 0.3
nu1 = 0.3

E0 = 1.0
E1 = 1.0

lamb0, mu0 = elut.youngPoisson2lame_planeStress(nu0, E0)
lamb1, mu1 = elut.youngPoisson2lame_planeStress(nu1, E1)

femData = {'fespace' : {'spaceType' : 'V', 'name' : 'u', 'spaceFamily' : 'CG', 'degree' : 2}}

# # Simulation
# traction = np.array([0.1,0.0])
# start = timer()
# for i in range(7,8):
#     print('running ', i)
#     MeshFile = "{0}{1}_{2}.{3}".format(folder,MeshFileRad, i, MeshFileExt)
#     mesh = fela.EnrichedMesh(MeshFile)
#     mesh.createFiniteSpace(**femData['fespace'])

#     paramAux = np.zeros((10,2))
#     getMapParamToDomain(paramAux, np.array([lamb0,lamb1,mu0,mu1]))
 
#     u = fela.solveElasticityBimaterial_twoBCs(paramAux, mesh, traction)
    
#     with HDF5File(MPI.comm_world, "{0}/solution_{1}_mpi.hdf5".format(folder,i), 'w') as f:
#         f.write(u, "u")
    
#     del mesh
#     del u
    
# end = timer()
# print(end - start) # Time in seconds, e.g. 5.38091952400282

# Loading 
# u = []
# meshList = []
# for i in range(Nrefine + 1):
#     print('reading ', i)
#     MeshFile = "{0}{1}_{2}.{3}".format(folder,MeshFileRad, i, MeshFileExt)
#     mesh = fela.EnrichedMesh(MeshFile)
#     mesh.createFiniteSpace('V','u', 'CG', 2)
#     uu = Function(mesh.V['u'])
    
#     with HDF5File(MPI.comm_world, "{0}/solution_{1}_mpi.hdf5".format(folder,i), 'r') as f:
#         f.read(uu, 'u')
        
#     u.append(uu)
#     meshList.append(mesh)



error = []
errorH10 = []
for i in range(Nrefine):
    
    u_ = interpolate(u[i], meshList[-1].V['u'])
    error = dot(u[-1]-u_,u[-1]-u_)*dx(mesh[-1].V('u'))
    E2 = sqrt(assemble(error))
    print('computing error L2', i)  
    error.append(errornorm(u[-1], u[i], norm_type='L2', degree_rise=3))
    # print(error[-1])
    # print('computing error H10', i)  
    # errorH10.append(errornorm(u[-1], u[i] , norm_type='H10', degree_rise=3))
    # print(errorH10[-1])

    
h = np.array([0.05/(2**i) for i in range(Nrefine)])

r = 1
Epred2 = error[0]*(h[-1]/h[0])**1
r = 1
Epred1 = errorH10[0]*(h[-1]/h[0])**r

rates = [ np.log(error[i]/error[i-1])/np.log(h[i]/h[i-1]) for i in range(1,Nrefine)]
print(rates)
print(error)

# rates = [ np.log(errorH10[i]/errorH10[i-1])/np.log(h[i]/h[i-1]) for i in range(1,Nrefine)]
# print(rates)
# print(errorH10)


# plt.figure(1)
# plt.plot(-np.log10(h),error,'-x')
# plt.plot(-np.log10(np.array([h[0],h[-1]])),[error[0],Epred2],'-o')
# plt.yscale('log')
# plt.xlabel('-log_{10} h')
# plt.ylabel('error L2 u')
# plt.grid()
# plt.savefig('convergence_u_L2_P1.png')

# plt.figure(2)
# plt.plot(-np.log10(h),errorH10,'-x')
# plt.plot(-np.log10(np.array([h[0],h[-1]])),[errorH10[0],Epred1],'-o')
# plt.yscale('log')
# plt.xlabel('-log_{10} h')
# plt.ylabel('error H10 u')
# plt.grid()
# plt.savefig('convergence_u_H10_P1.png')
# plt.show()

# plt.show()



# Computing errors in stresses
# param = 9*[(lamb0, mu0)] + [(lamb1, mu1)]
# i = Nrefine
# print('computing reference ')    
# lameRef = fela.myCoeff(meshList[i].subdomains, param, degree = 1)
# sigmaRef = lambda u: fela.sigmaLame(u,lameRef)
# Uref = Function(meshList[i].V['u'])
# Uref.assign(u[i])

# VsigRef = TensorFunctionSpace(meshList[i], "DG", degree=0)
# sig_ref = Function(VsigRef, name="Stress")
# sig_ref.assign(local_project(sigmaRef(Uref), VsigRef))

# sRef = sigmaRef(Uref) - (1./3)*tr(sigmaRef(Uref))*Identity(2)
# von_Mises_ref_ = sqrt(((3./2)*inner(sRef, sRef))) 

# Vsig0_ref = FunctionSpace(meshList[i], "DG", 0)
# von_Mises_ref = Function(Vsig0_ref, name="Stress")
# von_Mises_ref.assign(local_project(von_Mises_ref_, Vsig0_ref))

# error = []    
# errorMises = []
# for i in range(Nrefine):
#     print('computing error ', i)    
#     lame = fela.myCoeff(meshList[i].subdomains, param, degree = 1)
#     sigma = lambda u: fela.sigmaLame(u,lame)
#     Ui = Function(meshList[i].V['u'])
#     Ui.assign(u[i])
    
#     Vsig = TensorFunctionSpace(meshList[i], "DG", degree=0)
#     sig_i = Function(Vsig, name="Stress")
#     sig_i.assign(local_project(sigma(Ui), Vsig))
    
#     s = sigma(u[i]) - (1./3)*tr(sigma(u[i]))*Identity(2)
#     von_Mises = sqrt(((3./2)*inner(s, s))) 
    
#     Vsig0 = FunctionSpace(meshList[i], "DG", 0)
#     von_Mises_i = Function(Vsig0, name="Stress")
#     von_Mises_i.assign(local_project(von_Mises, Vsig0))
    
#     E =  errornorm(sig_ref, sig_i, norm_type='L2', degree_rise=0) # integrate over the mesh of the second argument
#     Emises = errornorm(von_Mises_ref, von_Mises_i, norm_type='L2', degree_rise=0) # integrate over the mesh of the second argument
    
#     error.append(E)
#     errorMises.append(Emises)

    
# h = np.array([0.05/(2**i) for i in range(Nrefine)])

# r = 2
# Epred2 = error[0]*(h[-1]/h[0])**r
# r = 1
# Epred1 = error[0]*(h[-1]/h[0])**r

# r = 1
# EpredMises1 = errorMises[0]*(h[-1]/h[0])**r

# rates = [ np.log(error[i]/error[i-1])/np.log(h[i]/h[i-1]) for i in range(1,Nrefine)]
# print(rates)

# rates = [ np.log(errorMises[i]/errorMises[i-1])/np.log(h[i]/h[i-1]) for i in range(1,Nrefine)]
# print(rates)

# plt.figure(1)
# plt.plot(-np.log10(h),error,'-x')
# plt.plot(-np.log10(np.array([h[0],h[-1]])),[error[0],Epred1],'-o')
# plt.yscale('log')
# plt.xlabel('-log_{10} h')
# plt.ylabel('error L2 sigma')
# plt.grid()
# plt.savefig('convergence_sigma_L2_P1.png')


# plt.figure(2)
# plt.plot(-np.log10(h),errorMises,'-x')
# plt.plot(-np.log10(np.array([h[0],h[-1]])),[errorMises[0],EpredMises1],'-o')
# plt.yscale('log')
# plt.xlabel('-log_{10} h')
# plt.ylabel('error L2 mises')
# plt.grid()
# plt.savefig('convergence_mises_L2_P1.png')

# plt.show()

