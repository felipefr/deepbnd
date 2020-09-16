import numpy as np
import sys, os
sys.path.insert(0, '../utils/')
from timeit import default_timer as timer
import elasticity_utils as elut
import ioFenicsWrappers as iofe
import Snapshots as snap
import Generator as gene
import genericParam as gpar
   
folderRB = "rbdata/"
folderSimuls = "simuls/"
Folder = 'mesh/'
MeshFile = Folder + 'mesh.xdmf'
referenceGeo = Folder + 'reference.geo'
SolutionFile = folderSimuls + "solution.hdf5"
DataFile = folderSimuls + "data_stress_Train_1500-2000.hdf5"
posProcFile = folderSimuls + "posProc.hdf5"

traction = np.array([0.0,0.05])

nparam = 5
nsTrain = 10000 # 10000
# nsRB = 10 # maybe tests parameters are screwed up because of this. Test with and without
nsTest = 1000 # 1000

pm = gpar.randomParams(seed=6, fileName = folderSimuls + 'paramFile.hdf5') # seed=6 
pm.addVariable('tx',-0.0625,0.0625)
pm.addVariable('ty',-0.0625,0.0625)
pm.addVariable('lx',-0.25,0.25)
pm.addVariable('ly',-0.25,0.25)
pm.addVariable('gamma',0.001,0.1)

# equivalent if started with seed 6 and 10000,10, 1000 as the sizes of samples
# pm.addSample('Train', nsTrain)
# pm.addSample('RB', nsRB)
# pm.addSample('Test', nsTest)

pm.addSample('Train', nsTrain, 6, 0)
# pm.addSample('RB', nsRB, 6, 50000 )
pm.addSample('Test', nsTest, 6 , 50000)

pm.write(mode = 'w')

pc = gpar.genericParams()
pc.addVariable('nu', 0.3)
pc.addVariable('E', 1.0)

pd = gpar.derivedParams(pm,pc)
pd.addVariable('lamb1', None, ['nu','E'] , elut.eng2lambPlane)
pd.addVariable('lamb2', ['gamma'],['nu','E'] , lambda x,y,z: elut.eng2lambPlane(y,x*z))
pd.addVariable('mu1', None, ['nu','E'] , elut.eng2mu)
pd.addVariable('mu2', ['gamma'],['nu','E'] , lambda x,y,z: elut.eng2mu(y,x*z))



# Creating reference Mesh
defaultMeshParam = iofe.getDefaultParameters()
d = defaultMeshParam
hRef = 0.005 # 0.005
Nref = int(d['Lx1']/hRef) + 1
hRef = d['Lx1']/(Nref - 1)
d['lc'] = hRef
d['Nx1'] = Nref
d['Nx2'] = 2*Nref - 1
d['Nx3'] = Nref
d['Ny1'] = Nref
d['Ny2'] = 2*Nref - 1
d['Ny3'] = Nref
    
# iofe.exportMeshXDMF_fromReferenceGeo(defaultMeshParam, referenceGeo , MeshFile)
# os.system("rm mesh.msh head.geo modifiedReference.geo")

s = snap.Snapshots(MeshFile, SolutionFile , posProcFile, pm, pd, defaultMeshParam, traction)

# s.resetHDF5file()
# s.writeBasics()
                  
# start = timer()

# # Simulation
# s.buildSnapshots('Train')
# s.buildSnapshots('Test')

# s.closeHDF5file(compress = False)
# posProcessingStress(self, label, indexes = [], posProcFile = None):
# end = timer()
# print(end - start) # Time in seconds, e.g. 5.38091952400282


# s.closeHDF5file(compress = False)

s.setDataset(DataFile, ['Train', 'Test'])
    
s.generateData('Train', 'Y', gene.stressGenerator( 10, 10,  1500, 2000), solutionFile = posProcFile)
# s.generateData('Train', 'X', gene.displacementGenerator(['Right','Bottom','Top'],[10,10,10], 0.05, 8000, 10000))

# # Posprocessing
# hdf5file = HDF5File(MPI.comm_world, SolutionFile, 'r')
# uu = Function(Mesh.V['u'])
# hdf5file.read(uu, 'basic')
# hdf5file.close()

# hdf5file = h5py.File(SolutionFile, 'a')
# for i in [1,4,7]:
#     print("posprocessing FEA %d"%(i))
#     p = pd(i,'FEA')
#     paramAux = np.array( 9*[[p[0] , p[2]]] + [[p[1] , p[3]]] )
#     iofe.moveMesh(Mesh, MeshRef, pm(i,'FEA',['tx','ty','lx','ly']), defaultMeshParam)
#     uu.vector().set_local(hdf5file['FEA/sol'][:,i])
#     # iofe.postProcessing_simple(uu, "{0}/output_{1}.xdmf".format(folderSimuls,i))
#     iofe.postProcessing_complete(uu, "{0}/output_{1}.xdmf".format(folderSimuls,i), labels = ['u','vonMises', 'lame'], param = paramAux)
    
# hdf5file.close()

# for i in range(nsTest):
#     print("building snapshot Test %d"%(i))
#     # snapshotsTest[:,i] = solveElasticityBarFenics(fespace,paramTest[i,:],angleGravity, isLame)
#     snapshotsTest[:,i] = solveElasticityBarFenics(fespace,paramTest[i,:], isLame, folder + 'displacement_' + str(i) + '.pvd')


# ========  RB approximation ==============================================
# tol = 1.0e-7

# U, sigma, ZT = np.linalg.svd(snapshotsFEA, full_matrices=False )
# print(sigma)

# N = 0
# sigma2_acc = 0.0
# threshold = (1.0 - tol*tol)*np.sum(sigma*sigma)
# while sigma2_acc < threshold and N<nsFEA:
#     sigma2_acc += sigma[N]*sigma[N]
#     N += 1  

# print(N)
# input()
# Vbase = U[:,:N]

# affineDecomposition = getAffineDecompositionElasticityBarFenics(fespace,Vbase)

# snapshotsRB = np.zeros((Nh,nsRB)) 
# for i in range(nsRB):
#     print("building snapshots RB %d"%(i))
#     snapshotsRB[:,i] = computeRBapprox(paramRB[i,:],affineDecomposition,Vbase) # theta is same param

# snapshotsRB_fea = np.zeros((Nh,nsRB)) 
# for i in range(nsRB):
#     print("building snapshots RB %d"%(i))
#     snapshotsRB_fea[:,i] = solveElasticityBarFenics(fespace,paramRB[i,:], isLame)

# errors = np.linalg.norm(snapshotsRB - snapshotsRB_fea,axis=0)
# print(errors)
# print(np.mean(errors))

# -------------------------------------------
# Saving

# np.savetxt(folder + "snapshotsFEA.txt",snapshotsFEA)        
# np.savetxt(folder + "snapshotsTest.txt",snapshotsTest)
# np.savetxt(folder + "snapshotsRB.txt",snapshotsRB)
# np.savetxt(folder + "paramFEA.txt",paramFEA)
# np.savetxt(folder + "paramTest.txt",paramTest)
# np.savetxt(folder + "paramRB.txt",paramRB)
# np.savetxt(folder + "paramLimits.txt",paramLimits)
# np.savetxt(folder + "U.txt",U)
# # np.savetxt(folder + "nodes.txt", fespace.mesh().coordinates()) # the mesh coordinates doesn't correspond to the nodes that stores dofs  
# np.savetxt(folder + "nodes.txt", fespace.tabulate_dof_coordinates()[0::2,:] )  # we need to jump because there are repeated coordinates
# np.savetxt(folder + "paramTest_poisson.txt",elut.convertParam2(paramTest[:,0:2], elut.composition(elut.lame2youngPoisson, elut.lameStar2lame)) )

# for j in range(2):
#     label = "ANq" + str(j)
#     np.savetxt(folder + label + ".txt", affineDecomposition[label])
    
#     label = "Aq" + str(j)        
#     np.savetxt(folder + label + ".txt", affineDecomposition[label])

# for j in range(2):
#     label = "fNq" + str(j)
#     np.savetxt(folder + label + ".txt", affineDecomposition[label])
    
#     label = "fq" + str(j)
#     np.savetxt(folder + label + ".txt", affineDecomposition[label])

# ==============================================================================

# tolList = [0.0, 1.e-8,1.e-6,1.e-4]

# for tol in tolList:
#     len(sigma)
#     if(tol<1.e-26):
#         N = ns
#     else:
#         sigma2_acc = 0.0
#         threshold = (1.0 - tol*tol)*np.sum(sigma*sigma)
#         N = 0
#         while sigma2_acc < threshold and N<ns:
#             sigma2_acc += sigma[N]*sigma[N]
#             N += 1  
  
#     V = U[:,:N]
    
#     affineDecomposition = getAffineDecompositionElasticityBarFenics(fespace,V)
    
#     tolEff = np.sqrt(1.0 - np.sum(sigma[0:N]*sigma[0:N])/np.sum(sigma*sigma))
    
#     errors = np.zeros(nsTest)
#     for i in range(nsTest):
#         nu =  paramTest[i,0]
#         E = paramTest[i,1] 
#         lamb = nu * E/((1. - 2.*nu)*(1.+nu))
#         mu = E/(2.*(1. + nu))
        
#         uR = computeRBapprox([lamb,mu],affineDecomposition)
                    
#         u = solveElasticityBarFenics(fespace,paramTest[i,:]) 
        
#         errors[i] = np.linalg.norm(uR - u)

#     print(" ====== Summary test RB for randomly sampled tested parameters (ns=%d,N=%d,eps=%d,epsEff) ==== ",ns,N,tol,tolEff)
#     print("errors = ", errors)
#     print("avg error = ", np.average(errors))
#     print("================================================================\n") 

