from __future__ import print_function
import numpy as np
from fenics import *
from dolfin import *
from ufl import nabla_div
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, '..')
import copy

import elasticity_utils as elut

import fenicsWrapperElasticity as fela 
from timeit import default_timer as timer

# metadata={"quadrature_degree": 1} 
def local_project(v,V):
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv,v_)*dx 
    b_proj = inner(v,v_)*dx
    solver = LocalSolver(a_proj,b_proj) 
    solver.factorize()
    u = Function(V)
    solver.solve_local_rhs(u)
    return u

def compute_errors(u_ref, u):
    """Compute various measures of the error u - u_ref, where
    u is a finite element Function and u_ref is an Expression."""

    # Get function space
    V = u.function_space()
    V_ref = u_ref.function_space()

    # Explicit computation of L2 norm
    error = dot(u - u_ref,u - u_ref)*dx(V_ref.mesh())
    E1 = sqrt(assemble(error))

    # Explicit interpolation of u_ref onto the same space as u
    u_ref_ = interpolate(u_ref, V)
    error = dot(u - u_ref_,u-u_ref)*dx(V.mesh())
    E2 = sqrt(assemble(error))

    # Explicit interpolation of u_ref to higher-order elements.
    #u will also be interpolated to the space Ve before integration
    Ve = VectorFunctionSpace(V.mesh(), 'P', 5)
    u_ref_ = interpolate(u_ref, Ve)
    u_ = interpolate(u, Ve)
    error = dot(u_ - u_ref_, u_ - u_ref_)*dx(Ve.mesh())
    E3 = sqrt(assemble(error))

    # # Infinity norm based on nodal values
    u_ref_ = interpolate(u_ref, V)
    E4 = np.abs(u_ref_.vector().get_local() - u.vector().get_local()).max()

    # L2 norm
    E5 = errornorm(u_ref, u, norm_type='L2', degree_rise=3)

    # H1 seminorm
    E6 = errornorm(u_ref, u, norm_type='H10', degree_rise=3)

    # Collect error measures in a dictionary with self-explanatory keys
    errors = {'u - u_ref': E1,
              'u - interpolate(u_ref, V)': E2,
               'interpolate(u, Ve) - interpolate(u_ref, Ve)': E3, 
               'infinity norm (of dofs)': E4,
               'L2 norm': E5,
               'H10 seminorm': E6}

    return errors

zeroParam = 4*[0.0]

MeshFile = 'mesh0.xml'
referenceGeo = 'reference.geo'

d = fela.getDefaultParameters()
hRef = 0.05
Nref = int(d['Lx1']/hRef) + 1
d['lc'] = hRef
d['Nx1'] = Nref
d['Nx2'] = 2*Nref - 1
d['Nx3'] = Nref
d['Ny1'] = Nref
d['Ny2'] = 2*Nref - 1
d['Ny3'] = Nref
    
# fela.exportMeshXML(d, referenceGeo, MeshFile)
# os.system("mv mesh.msh mesh0.msh")
# os.system("mv mesh0.msh meshTemp.msh")

Nrefine = 6
# for i in range(Nrefine):
#     print("refining {0}th times".format(i+1))
#     os.system("gmsh meshTemp.msh -refine -format 'msh2' -o meshTemp2.msh")
#     os.system("dolfin-convert meshTemp2.msh mesh{0}.xml".format(i+1))
#     os.system("mv meshTemp2.msh meshTemp.msh")
    


nu0 = 0.3
nu1 = 0.3

E0 = 1.0
E1 = 100.0

lamb0, mu0 = elut.youngPoisson2lame_planeStress(nu0, E0)
lamb1, mu1 = elut.youngPoisson2lame_planeStress(nu1, E1)

# Simulation
# start = timer()
# for i in range(Nrefine+1):
#     print('running ', i)
#     MeshFile = 'mesh{0}.xml'.format(i)
#     mesh = fela.EnrichedMesh(MeshFile)
#     u = fela.solveElasticityBimaterial(np.array(9*[[lamb0, mu0]] + [[lamb1, mu1]]), mesh)
#     del mesh
#     # fela.postProcessing(u, other[0], other[1] , other[2], "output_{0}.xdmf".format(i) )
#     # fela.postProcessing_simple(u, "output_{0}.xdmf".format(i))
    
#     hdf5file = HDF5File(MPI.comm_world, "solution{0}_P1.hdf5".format(i), 'w')
#     hdf5file.write(u, "u")
    
#     del u
    
# end = timer()
# print(end - start) # Time in seconds, e.g. 5.38091952400282

# Loading 
u = []
meshList = []
for i in range(Nrefine + 1):
    print('reading ', i)
    MeshFile = 'mesh{0}.xml'.format(i)
    mesh = fela.EnrichedMesh(MeshFile)
    mesh.createFiniteSpace('V','u', 'CG', 1)
    uu = Function(mesh.V['u'])
    hdf5file = HDF5File(MPI.comm_world, "solution{0}_P1.hdf5".format(i), 'r')
    hdf5file.read(uu, 'u')
    u.append(uu)
    meshList.append(mesh)



# error = []
# errorH10 = []
# for i in range(Nrefine):
#     print('computing error L2', i)  
#     error.append(errornorm(u[-1], u[i], norm_type='L2', degree_rise=3))
#     print(error[-1])
#     print('computing error H10', i)  
#     errorH10.append(errornorm(u[-1], u[i] , norm_type='H10', degree_rise=3))
#     print(errorH10[-1])


# h = np.array([0.05/(2**i) for i in range(Nrefine)])

# r = 1
# Epred2 = error[0]*(h[-1]/h[0])**1
# r = 1
# Epred1 = errorH10[0]*(h[-1]/h[0])**r

# rates = [ np.log(error[i]/error[i-1])/np.log(h[i]/h[i-1]) for i in range(1,Nrefine)]
# print(rates)
# print(error)

# rates = [ np.log(errorH10[i]/errorH10[i-1])/np.log(h[i]/h[i-1]) for i in range(1,Nrefine)]
# print(rates)
# print(errorH10)

# # 
# # error u L2 = [1.627327209787531, 1.6084080743532907, 1.6223367414776917, 1.6897861093053061, 1.976734575042753]
# # rate u L2 = [6.4565681466076e-05, 2.0899092890473685e-05, 6.854067239981565e-06, 2.2262623330930235e-06, 6.90080941356396e-07, 1.7532491324848473e-07]

# plt.figure(1)
# plt.plot(-np.log10(h),error,'-x')
# plt.plot(-np.log10(np.array([h[0],h[-1]])),[error[0],Epred2],'-o')
# plt.yscale('log')
# plt.xlabel('-log_{10} h')
# plt.ylabel('error L2 u')
# plt.grid()
# plt.savefig('convergence_u_L2_P1.png')

# # error u H10 = [0.83215543483859, 0.8023366898771523, 0.7918877846722221, 0.7763717496402137, 1.1060081026646678]
# # rate u L2 = [0.004374250298529197, 0.002456970166806748, 0.0014088750328483115, 0.0008137490536819918, 0.00047509374444211245, 0.00022071797232863766]
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

# Results for P1
# [1.5001697885726464, 1.5659046479754304, 1.6096522065053704, 1.6921903477992912, 1.989405879514768]
# [0.0006465059086496437, 0.00022854745705820796, 7.719552480725472e-05, 2.529522386597343e-05, 7.827777530067249e-06, 1.9713676747050123e-06]
# [0.8300093877369954, 0.837586309280721, 0.8404141082759806, 0.8575485707157251, 0.8577474777358145]
# [0.01574917896807685, 0.00885931606396238, 0.004957487099552497, 0.002768673458229305, 0.0015280019020811266, 0.0008431719985780568]


# Computing errors in stresses
param = 9*[(lamb0, mu0)] + [(lamb1, mu1)]
i = Nrefine
print('computing reference ')    
lameRef = fela.myCoeff(meshList[i].subdomains, param, degree = 1)
sigmaRef = lambda u: fela.sigmaLame(u,lameRef)
Uref = Function(meshList[i].V['u'])
Uref.assign(u[i])

VsigRef = TensorFunctionSpace(meshList[i], "DG", degree=0)
sig_ref = Function(VsigRef, name="Stress")
sig_ref.assign(local_project(sigmaRef(Uref), VsigRef))

sRef = sigmaRef(Uref) - (1./3)*tr(sigmaRef(Uref))*Identity(2)
von_Mises_ref_ = sqrt(((3./2)*inner(sRef, sRef))) 

Vsig0_ref = FunctionSpace(meshList[i], "DG", 0)
von_Mises_ref = Function(Vsig0_ref, name="Stress")
von_Mises_ref.assign(local_project(von_Mises_ref_, Vsig0_ref))

error = []    
errorMises = []
for i in range(Nrefine):
    print('computing error ', i)    
    lame = fela.myCoeff(meshList[i].subdomains, param, degree = 1)
    sigma = lambda u: fela.sigmaLame(u,lame)
    Ui = Function(meshList[i].V['u'])
    Ui.assign(u[i])
    
    Vsig = TensorFunctionSpace(meshList[i], "DG", degree=0)
    sig_i = Function(Vsig, name="Stress")
    sig_i.assign(local_project(sigma(Ui), Vsig))
    
    s = sigma(u[i]) - (1./3)*tr(sigma(u[i]))*Identity(2)
    von_Mises = sqrt(((3./2)*inner(s, s))) 
    
    Vsig0 = FunctionSpace(meshList[i], "DG", 0)
    von_Mises_i = Function(Vsig0, name="Stress")
    von_Mises_i.assign(local_project(von_Mises, Vsig0))
    
    E =  errornorm(sig_ref, sig_i, norm_type='L2', degree_rise=0) # integrate over the mesh of the second argument
    Emises = errornorm(von_Mises_ref, von_Mises_i, norm_type='L2', degree_rise=0) # integrate over the mesh of the second argument
    
    error.append(E)
    errorMises.append(Emises)

    
h = np.array([0.05/(2**i) for i in range(Nrefine)])

r = 2
Epred2 = error[0]*(h[-1]/h[0])**r
r = 1
Epred1 = error[0]*(h[-1]/h[0])**r

r = 1
EpredMises1 = errorMises[0]*(h[-1]/h[0])**r

rates = [ np.log(error[i]/error[i-1])/np.log(h[i]/h[i-1]) for i in range(1,Nrefine)]
print(rates)

rates = [ np.log(errorMises[i]/errorMises[i-1])/np.log(h[i]/h[i-1]) for i in range(1,Nrefine)]
print(rates)


# errorSigma= [0.00187522, 0.00117677, 0.0006716 , 0.00038525, 0.00021625, 0.00010727]
# ratesSigma= [0.6722285498990327, 0.809152254062119, 0.8018082628430737, 0.8331046680054719, 1.0113877403592757]

plt.figure(1)
plt.plot(-np.log10(h),error,'-x')
plt.plot(-np.log10(np.array([h[0],h[-1]])),[error[0],Epred1],'-o')
plt.yscale('log')
plt.xlabel('-log_{10} h')
plt.ylabel('error L2 sigma')
plt.grid()
plt.savefig('convergence_sigma_L2_P1.png')

# errorMises = [1.61659626e-03, 8.99784302e-04, 5.11478672e-04, 2.93885594e-04, 1.64216333e-04, 8.40828375e-05]    
# ratesMises = [0.8453083141523603, 0.814905115068547, 0.7994194421628462, 0.8396570128734544, 0.9657143653816982]
plt.figure(2)
plt.plot(-np.log10(h),errorMises,'-x')
plt.plot(-np.log10(np.array([h[0],h[-1]])),[errorMises[0],EpredMises1],'-o')
plt.yscale('log')
plt.xlabel('-log_{10} h')
plt.ylabel('error L2 mises')
plt.grid()
plt.savefig('convergence_mises_L2_P1.png')

plt.show()

# Results for P1
# [0.5944629507653771, 0.8318960362433566, 0.48616598311544734, 1.3398958964535916, -0.1606309341266525]
# array([0.00917237, 0.00607477, 0.00341275, 0.00243643, 0.00096251,
#        0.00107587])
# [0.7721411809600918, 0.8817500189036049, 0.5347216344116275, 1.3035488100714034, -0.09378549081975819]
# array([0.0062262 , 0.00364574, 0.00197857, 0.00136579, 0.00055332,
#        0.00059049])