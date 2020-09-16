import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0,'../')
sys.path.insert(0,'../../utils/')
import matplotlib.pyplot as plt
import numpy as np
from dolfin import *
import myTensorflow as mytf
from timeit import default_timer as timer
import fenicsWrapperElasticity as fela
import elasticity_utils as elut
import fenicsMultiscale as fmts
import generation_deepBoundary_lib as gdb

import h5py
import pickle
import Generator as gene
import myHDF5 
import dataManipMisc as dman 
from sklearn.preprocessing import MinMaxScaler
import miscPosprocessingTools as mpt
import multiphenicsMultiscale as mpms

import json
from mpi4py import MPI as pyMPI
import myHDF5 
from basicTraining_lib import *

comm = MPI.comm_world


from basicTraining_lib import *

dotProductL2bnd = lambda u,v, m : assemble(inner(u,v)*ds)
dotProductH10 = lambda u,v, dx : assemble(inner(grad(u),grad(v))*dx)
dotProductL2 = lambda u,v, dx : assemble(inner(u,v)*dx) 

simul_id = 3
EpsDirection = 0
base_offline_folder = "/Users/felipefr/EPFL/newDLPDES/DATA/deepBoundary/data{0}/".format(simul_id)

arch_id = 1

nX = 4 # because the only first 4 are relevant, the other are constant

print("starting with", simul_id, EpsDirection, arch_id)


# Arch 1
net = {'Neurons': 5*[128], 'drps': 7*[0.0], 'activations': ['relu','relu','relu'], 
        'reg': 0.00001, 'lr': 0.01, 'decay' : 0.5, 'epochs': 1000}

# Arch 2
# net = {'Neurons': [128,256,256,256,512,1024], 'drps': 8*[0.1], 'activations': ['relu','relu','relu'], 
#         'reg': 0.0001, 'lr': 0.001, 'decay' : 0.5, 'epochs': 1000}


fnames = {}      

# ### fnames['prefix_out'] = 'saves_PODMSE_comparisonLessNy_Eps{0}_arch{1}/'.format(EpsDirection, arch_id)
# fnames['prefix_out'] = 'saves_correct_PODMSE_L2bnd_Eps{0}_arch{1}/'.format(EpsDirection, arch_id)
fnames['prefix_out'] = 'saves_correct_PODMSE_H10_Eps{0}_arch{1}/'.format(EpsDirection, arch_id)
fnames['prefix_in_X'] = base_offline_folder + "ellipseData_{0}.txt"
# fnames['prefix_in_Y'] = "./definitiveBasis/Y_L2bnd_original_{0}_{1}.hd5".format(simul_id, EpsDirection)
fnames['prefix_in_Y'] = "./definitiveBasis/Y_H10_lite2_correction_{0}_{1}.hd5".format(simul_id, EpsDirection)

ns = 1000
nsTrain = int(ns)

# nYlist = np.array([2,5,10,20,50,80,110,150]).astype('int')
nYlist = np.array([2,5,8,12,16,20,25,30,35,40]).astype('int')
Nruns = 2

models = []

Neurons = net['Neurons']
actLabel = net['activations']
drps = net['drps']
reg = net['reg']
Epochs = net['epochs']
decay = net['decay']
lr = net['lr']


X = []; Y =[]; scalerX = []; scalerY =[]

Xj, Yj, scalerXj, scalerYj = getTraining(0,nsTrain, nX, np.max(nYlist), fnames['prefix_in_X'], fnames['prefix_in_Y'])
X.append(Xj); Y.append(Yj);  scalerX.append(scalerXj); scalerY.append(scalerYj) 

models = []
for i, nY in enumerate(nYlist):
    models.append([])
    for j in range(Nruns):
        fnames['suffix_out'] = '_{0}_{1}'.format(nY,j)        
        models[-1].append(mytf.DNNmodel(nX, nY, Neurons, actLabel = actLabel, drps = drps, lambReg = reg))
        models[-1][-1].load_weights( fnames['prefix_out'] + 'weights' + fnames['suffix_out'])
        

# nameMeshRef = base_offline_folder + "RVE_POD_reduced_{0}.{1}".format(0,'xml')

# meshRef = refine(fela.EnrichedMesh(nameMeshRef))
# Vref = VectorFunctionSpace(meshRef,"CG", 1)

nx = 100
meshRef = RectangleMesh(Point(1.0/3., 1./3.), Point(2./3., 2./3.), nx, nx, diagonal='crossed')
Vref = VectorFunctionSpace(meshRef,"CG", 2)

dxRef = Measure('dx', meshRef) 
dsRef = Measure('ds', meshRef) 

# nameInterpolation = 'interpolatedSolutions_{0}_{1}.txt'
nameInterpolation = './definitiveBasis/interpolatedSolutions_lite2_{0}_{1}.hd5'
Isol = myloadfile(nameInterpolation.format(simul_id,EpsDirection),'Isol')

nameWbasis = './definitiveBasis/Wbasis_{0}{1}_{2}.hd5'
nameYlist = './definitiveBasis/Y_{0}{1}_{2}.hd5'
nameStress = base_offline_folder + 'sigmaList{0}.txt'
nameTau = './definitiveBasis/tau_H10_lite2_correction_3_0.hd5'
# nameTau = './definitiveBasis/tau_L2bnd_original_solvePDE_complete_3_0.hd5'

Nmax = 40 # 150 
ntest = 100 # 100
Wbasis = myloadfile(nameWbasis.format('H10_lite2_correction_', simul_id,EpsDirection),'Wbasis')[:Nmax,:]
Ylist = myloadfile(nameYlist.format('H10_lite2_correction_', simul_id,EpsDirection),'Ylist')
# Wbasis = myloadfile(nameWbasis.format('L2bnd_original_', simul_id,EpsDirection),'Wbasis')[:Nmax,:]
# Ylist = myloadfile(nameYlist.format('L2bnd_original_', simul_id,EpsDirection),'Ylist')
# tau = np.loadtxt(nameTau.format(simul_id,EpsDirection))
# tau0 = np.loadtxt(nameTau.format(simul_id,'fixedEps' + str(EpsDirection)))

tau, tau0 = myhd.loadhd5(nameTau,['tau','tau_0'])
sigmaList = np.loadtxt(nameStress.format(EpsDirection))[:ntest,[0,3,1]] # flatten to voigt notation
        
mseErrors = {}
mseErrors['POD_norm_stress'] = gdb.getMSEstresses(nYlist,Ylist[:ntest,:],tau,tau0,sigmaList)
mseErrors['POD_norm_L2bnd'] = gdb.getMSE(nYlist,Ylist[:ntest,:],Wbasis, Isol[:ntest,:], Vref, dsRef, dotProductL2bnd)
mseErrors['POD_norm_H10'] = gdb.getMSE(nYlist,Ylist[:ntest,:],Wbasis, Isol[:ntest,:], Vref, dxRef, dotProductH10)
mseErrors['POD_norm_L2'] = gdb.getMSE(nYlist,Ylist[:ntest,:],Wbasis, Isol[:ntest,:], Vref, dxRef, dotProductL2)


errors = {}
errors['L2bnd'] = []
errors['H10'] = []
errors['L2'] = []
errors['stress'] = []

run = 0
for i, nY in enumerate(nYlist):
    Yp_bar = models[i][run].predict(X[0][:ntest])
    Yp = scalerY[0].inverse_transform(np.concatenate((Yp_bar, np.zeros((ntest,Nmax-nY))),axis = 1))[:,:nY]
    errors['stress'].append(gdb.getMSEstresses([nY],Yp,tau,tau0,sigmaList)[0])
    errors['L2bnd'].append(gdb.getMSE([nY],Yp,Wbasis, Isol[:ntest,:], Vref, dsRef, dotProductL2bnd)[0])
    errors['H10'].append(gdb.getMSE([nY],Yp,Wbasis, Isol[:ntest,:], Vref, dxRef, dotProductH10)[0])
    errors['L2'].append(gdb.getMSE([nY],Yp,Wbasis, Isol[:ntest,:], Vref, dxRef, dotProductL2)[0])
    
    
mseErrors['model_norm_L2bnd'] = np.array(errors['L2bnd']) 
mseErrors['model_norm_H10'] = np.array(errors['H10']) 
mseErrors['model_norm_L2'] = np.array(errors['L2']) 
mseErrors['model_norm_stress'] = np.array(errors['stress']) 


plt.figure(3, (9,7))
plt.suptitle("Average errors 100 snapshots (H10-based RB), Arch 1, Seed {0}".format(run+1))
plt.subplot('221')
plt.title('$L^2(\partial \Omega_{\mu})$')
plt.plot(nYlist,mseErrors['POD_norm_L2bnd'], '-o')
plt.plot(nYlist,mseErrors['model_norm_L2bnd'], '-o')
plt.xlabel('N')
plt.ylabel('error sqrt(mse)')
plt.yscale('log')
plt.grid()
plt.legend(['POD', 'DNN'])
plt.subplot('222')
plt.title('$L^2(\Omega_{\mu})$')
plt.plot(nYlist,mseErrors['POD_norm_L2'], '-o')
plt.plot(nYlist,mseErrors['model_norm_L2'], '-o')
plt.xlabel('N')
plt.ylabel('error sqrt(mse)')
plt.yscale('log')
plt.grid()
plt.legend(['POD', 'DNN'])
plt.subplot('223')
plt.title('$H^1_0(\Omega_{\mu})$')
plt.plot(nYlist,mseErrors['POD_norm_H10'], '-o')
plt.plot(nYlist,mseErrors['model_norm_H10'], '-o')
plt.xlabel('N')
plt.ylabel('error sqrt(mse)')
plt.yscale('log')
plt.grid()
plt.legend(['POD', 'DNN'])
plt.subplot('224')
plt.title('Homogenised Stress')
plt.plot(nYlist,mseErrors['POD_norm_stress'], '-o')
plt.plot(nYlist,mseErrors['model_norm_stress'], '-o')
plt.xlabel('N')
plt.ylabel('error sqrt(mse)')
plt.yscale('log')
plt.grid()
plt.legend(['POD', 'DNN']) 
plt.tight_layout(rect=[0, 0.03, 1, 0.97])


plt.savefig("comparisonPODMSE/norms_POD_DNN_H10_corrected_arch1_seed{0}.png".format(run+1))
plt.show()

# ========= END OF THE ORIGINAL PROGRAM ====================

# contrast = 10.0
# E2 = 1.0
# E1 = contrast*E2 # inclusions
# nu1 = 0.3
# nu2 = 0.3

# mu1 = elut.eng2mu(nu1,E1)
# lamb1 = elut.eng2lambPlane(nu1,E1)
# mu2 = elut.eng2mu(nu2,E2)
# lamb2 = elut.eng2lambPlane(nu2,E2)

# param = np.array([[lamb1, mu1], [lamb2,mu2],[lamb1, mu1], [lamb2,mu2]])

 
# EpsUnits = [np.array([[1.,0.],[0.,0.]]), np.array([[0.,0.],[0.,1.]]), np.array([[0.,0.5],[0.5,0.]])]

# EpsFluc = []
# for j in range(3):
#     EpsFluc.append(np.loadtxt(base_offline_folder + 'EpsList_{0}.txt'.format(j)))

# Ntest = 2
# sigma_hat = np.zeros((Ntest,4))
# sigma_bar = np.zeros((Ntest,4))
# Urec = []

# for nt in range(Ntest):
#     print(".... Now computing test ", nt)
#     # Generating reference solution
#     ellipseData = np.loadtxt(base_offline_folder + 'ellipseData_' + str(nt) + '.txt')[:nX,2]
#     nameMesh = base_offline_folder + "RVE_POD_reduced_{0}.{1}".format(nt,'xml')
#     mesh = fela.EnrichedMesh(nameMesh)
#     V = VectorFunctionSpace(mesh,"CG", 1)

#     # for this case
#     epsL = EpsUnits[0] + EpsFluc[0][nt,:].reshape((2,2))
#     sigmaL, sigmaEpsL = fmts.getSigma_SigmaEps(param[0:2],mesh,epsL)
    
#     x_epsL = [epsL[0,0],epsL[1,1],0.5*(epsL[0,1] + epsL[1,0])]

#     v = Function(V)
#     vtemp = np.zeros(len(v.vector()[:]))
    
#     sigma_bar[nt,:] = fmts.homogenisation(v, mesh, sigmaL, [0,1], sigmaEpsL).flatten() # zero for v
    
#     sigmaL0, sigmaEpsL0 = fmts.getSigma_SigmaEps(param[0:2],mesh,np.zeros((2,2)))
#     for j in range(3):
#         X = scalerX[j].transform(ellipseData.reshape((1,nX)))
#         # Y = models[j].predict(X)
#         # alpha = scalerY[j].inverse_transform(Y)
#         alpha = Y[j][nt,:].reshape((1,nY))
        
#         for i in range(nY):
#             vtemp = vtemp + x_epsL[j]*alpha[0,i]*interpolate(Wbasis_func[j][i],V).vector()[:]
    
#     v.vector()[:] = vtemp
    
#     if(op_solveForInternal):
#         Ured = mpms.solveMultiscale(param[0:2,:], mesh, epsL, op = 'BCdirich_lag', others = [v])
#         sigma_hat[nt,:] = fmts.homogenisation(Ured[0], mesh, sigmaL, [0,1], sigmaEpsL).flatten()
#         Urec.append(Ured[0])
#     else:    
#         sigma_hat[nt,:] = sigma_bar[nt,:] + fmts.homogenisation(v, mesh, sigmaL0, [0,1], sigmaEpsL0).flatten()
#         Urec.append(v)


# print(sigma_hat)
# print(sigma_bar)

# nameStress = base_offline_folder + 'sigmaList{0}.txt'
# sigma_true = np.loadtxt(nameStress.format(0))[:Ntest,:]

# print(sigma_hat - sigma_true)

# print(np.linalg.norm(sigma_hat - sigma_true,axis = 1))

# =========================  Recomputing true sigmas =================================================
# EpsDirection = 0
# nameSol = base_offline_folder + 'RVE_POD_solution_red_{0}_' + str(EpsDirection) + '.{1}'
# nameMesh = base_offline_folder + "RVE_POD_reduced_{0}.{1}"

# Usol = [] 
# for nt in range(Ntest):
#     print(".... Now computing test ", nt)

#     mesh = fela.EnrichedMesh(nameMesh.format(nt,'xml'))
#     V = VectorFunctionSpace(mesh,"CG", 1)
    
#     u = Function(V)
    

#     # for this case
#     epsL = EpsUnits[0] + EpsFluc[0][nt,:].reshape((2,2))
#     sigmaL, sigmaEpsL = fmts.getSigma_SigmaEps(param[0:2],mesh,epsL)
    
#     sigma_bar = fmts.homogenisation(u, mesh, sigmaL, [0,1], sigmaEpsL).flatten()
    
#     with HDF5File(comm, nameSol.format(nt, 'h5') , 'r') as f:
#         f.read(u, 'basic')

#     sigmaL0, sigmaEpsL0 = fmts.getSigma_SigmaEps(param[0:2],mesh,np.zeros((2,2)))
#     sigma_hat[nt,:] = sigma_bar + fmts.homogenisation(u, mesh, sigmaL0, [0,1], sigmaEpsL0).flatten()
    
#     Usol.append(u)




# print(sigma_hat)


# sigma_hat = np.zeros((Ntest,4))
# # Computing tau_ji
# tau = []

# for nt in range(Ntest):
#     print(".... Now computing tau for test ", nt)
#     nameMesh = base_offline_folder + "RVE_POD_reduced_{0}.{1}".format(nt,'xml')
#     mesh = fela.EnrichedMesh(nameMesh)
#     V = VectorFunctionSpace(mesh,"CG", 1)

#     tau.append([])
#     for j in range(3):
#         epsL = EpsUnits[j] + EpsFluc[j][nt,:].reshape((2,2))
#         sigmaL, sigmaEpsL = fmts.getSigma_SigmaEps(param,mesh,epsL)
        
#         tau[-1].append([])
        
#         for i in range(nY):
#             Iwi = interpolate(Wbasis_func[j][i],V)
#             tau[-1][-1].append(fmts.homogenisation(Iwi, mesh, sigmaL, [0,1], sigmaEpsL).flatten())
    


# for nt in range(Ntest):
#     print(".... Now computing test ", nt)
#     # Generating reference solution
#     ellipseData = np.loadtxt(base_offline_folder + 'ellipseData_' + str(nt) + '.txt')[:nX,2]
    
#     for j in range(1):
#         alpha = models[j].predict(ellipseData.reshape((1,nX)))
#         for i in range(nY):
#             sigma_hat[nt,:] += alpha[0,i]*tau[nt][j][i] 


# print(sigma_hat)


# sigma_hat = np.zeros((Ntest,4))  
# Urec = Function(V)
# Urec.vector()[:] = 0.0
# alpha = models[EpsDirection].predict(ellipseData.reshape((1,nX)))
# for i in range(nY):
#     Urec.vector()[:] += alpha[0,i]*interpolate(Wbasis_func[i],V).vector()[:]

# sigma_hat[nt,:] = fmts.homogenisation(Urec, mesh, sigmaL, [0,1], sigmaEpsL).flatten()


# X_t, Y_t, scalerXdummy, scalerYdummy = getTraining(nsTrain,ns, scalerX, scalerY)


# Y_p = model.predict(X_t)



# ns = 400
# nsTrain = int(0.9*ns)
# nsTest = ns - nsTrain

# X, Y, scalerX, scalerY = getTraining(0,nsTrain)


# # Prediction step
# ntest = 2000

# X_t, Y_t, scalerXdummy, scalerYdummy = getTraining(nsTrain,ns, scalerX, scalerY)


# Y_p = model.predict(X_t)

# error = {}
# error["ml2"] = np.linalg.norm(Y_p - Y_t)/nsTest
# error["rl2"] = np.linalg.norm(Y_p - Y_t)/np.linalg.norm(Y_t)
# error["ml2_0"] = list(np.linalg.norm(Y_p - Y_t, axis = 0)/nsTest)
# error["rl2_0"] = list(np.linalg.norm(Y_p - Y_t, axis = 0)/np.linalg.norm(Y_t,axis = 0))

# Y_tt = scalerY.inverse_transform(Y_t)
# Y_pp = scalerY.inverse_transform(Y_p)

# j = 0 
# mpt.visualiseStresses9x9(Y_tt[j:j+9,0:3] , Y_pp[j:j+9,0:3] , 
#                       figNum = 1, savefig = partialRadical + '_twoPoints_dispBound_Min_{0}.png'.format(Run_id))

# mpt.visualiseStresses9x9( Y_tt[j:j+9,3:] , Y_pp[j:j+9,3:] , 
#                       figNum = 2, savefig = partialRadical + '_twoPoints_dispBound_Max_{0}.png'.format(Run_id))


# print(error)
# with open(partialRadical + '_twoPoints_dispBound_error_{0}.json'.format(Run_id), 'w') as file:
#      file.write(json.dumps(error)) # use `json.loads` to do the reverse


# mpt.visualiseScatterErrors(Y_t[:,0:3], Y_p[:,0:3], ['stress Min','x Min','y Min'], gamma = 1.0, 
#                            figNum = 3, savefig = partialRadical + '_scatterError_twoPoints_dispBound_Min_{0}.png'.format(Run_id))

# mpt.visualiseScatterErrors(Y_t[:,3:], Y_p[:,3:], ['stress Max','x Max','y Max'], gamma = 1.0, 
#                            figNum = 4, savefig = partialRadical + '_scatterError_twoPoints_dispBound_Max_{0}.png'.format(Run_id))

# plt.show()
    