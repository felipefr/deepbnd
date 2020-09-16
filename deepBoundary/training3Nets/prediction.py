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

comm = MPI.comm_world



# base_offline_folder = "/home/felipefr/EPFL/newDLPDEs/DATA/deepBoundary/data1/"
# Xdatafile = base_offline_folder + "ellipseData_{0}.txt"


# 7)
# Neurons= 6*[256]
# drps = 8*[0.0]
# lr2 = 0.00001
# lr = 0.01
# decay = 0.5

# 6)
Neurons= 5*[128]
drps = 7*[0.0]
lr2 = 0.00001
lr = 0.01
decay = 0.5


case_id = 6
simul_id = 3
nY = 1000
nX = 4

Nin = nX
Nout = nY

op_solveForInternal = False


base_offline_folder = "/Users/felipefr/EPFL/newDLPDES/DATA/deepBoundary/data{0}/".format(simul_id)
Xdatafile = base_offline_folder + "ellipseData_{0}.txt"
Ydatafile = "Y_corrected_{0}_{1}.txt"


def getTraining(ns_start, ns_end, Ydatafile, scalerX = None, scalerY = None):
    X = np.zeros((ns_end - ns_start,nX))
    Y = np.zeros((ns_end - ns_start,nY))
    
    for i in range(ns_end - ns_start):
        j = i + ns_start
        X[i,:] = np.loadtxt(Xdatafile.format(j))[:nX,2]
    
    Y = np.loadtxt(Ydatafile)[ns_start:ns_end,:nY] 
   

    if(type(scalerX) == type(None)):
        scalerX = MinMaxScaler()
        scalerX.fit(X)
    
    if(type(scalerY) == type(None)):
        scalerY = MinMaxScaler()
        scalerY.fit(Y)
            
    return scalerX.transform(X), scalerY.transform(Y), scalerX, scalerY


ns = 1000
nsTrain = int(ns)

X = []; Y =[]; scalerX = []; scalerY =[]
for j in range(3):
    Xj, Yj, scalerXj, scalerYj = getTraining(0,nsTrain, Ydatafile.format(simul_id, j))
    X.append(Xj); Y.append(Yj);  scalerX.append(scalerXj); scalerY.append(scalerYj) 

folderSavings = 'saves_{0}_{1}/'.format(simul_id,nY) 

# models = []
# for j in range(3): # EpsDirection
#     models.append(mytf.DNNmodel(Nin, Nout, Neurons, actLabel = ['relu','relu','relu'], drps = drps, lambReg = lr2  ))
#     models[-1].load_weights(folderSavings + 'weights_{0}_{1}'.format(j,case_id))


nameMeshRef = base_offline_folder + "RVE_POD_reduced_{0}.{1}".format(0,'xml')
meshRef = refine(fela.EnrichedMesh(nameMeshRef))
Vref = VectorFunctionSpace(meshRef,"CG", 1)
dxRef = Measure('dx', meshRef) 


nameWbasis = 'Wbasis_complete_{0}_{1}.txt'
EpsDirection = 0

Wbasis = []
Wbasis_func = []
for j in range(3):
    Wbasis.append(np.loadtxt(nameWbasis.format(simul_id,j))[:,:nY])
    
    Wbasis_func.append([])
    for i in range(nY):
        Wbasis_func[-1].append(Function(Vref))
        Wbasis_func[-1][-1].vector()[:] = Wbasis[-1][:,i]

   
contrast = 10.0
E2 = 1.0
E1 = contrast*E2 # inclusions
nu1 = 0.3
nu2 = 0.3

mu1 = elut.eng2mu(nu1,E1)
lamb1 = elut.eng2lambPlane(nu1,E1)
mu2 = elut.eng2mu(nu2,E2)
lamb2 = elut.eng2lambPlane(nu2,E2)

param = np.array([[lamb1, mu1], [lamb2,mu2],[lamb1, mu1], [lamb2,mu2]])

 
EpsUnits = [np.array([[1.,0.],[0.,0.]]), np.array([[0.,0.],[0.,1.]]), np.array([[0.,0.5],[0.5,0.]])]

EpsFluc = []
for j in range(3):
    EpsFluc.append(np.loadtxt(base_offline_folder + 'EpsList_{0}.txt'.format(j)))

Ntest = 2
sigma_hat = np.zeros((Ntest,4))
sigma_bar = np.zeros((Ntest,4))
Urec = []

for nt in range(Ntest):
    print(".... Now computing test ", nt)
    # Generating reference solution
    ellipseData = np.loadtxt(base_offline_folder + 'ellipseData_' + str(nt) + '.txt')[:nX,2]
    nameMesh = base_offline_folder + "RVE_POD_reduced_{0}.{1}".format(nt,'xml')
    mesh = fela.EnrichedMesh(nameMesh)
    V = VectorFunctionSpace(mesh,"CG", 1)

    # for this case
    epsL = EpsUnits[0] + EpsFluc[0][nt,:].reshape((2,2))
    sigmaL, sigmaEpsL = fmts.getSigma_SigmaEps(param[0:2],mesh,epsL)
    
    x_epsL = [epsL[0,0],epsL[1,1],0.5*(epsL[0,1] + epsL[1,0])]

    v = Function(V)
    vtemp = np.zeros(len(v.vector()[:]))
    
    sigma_bar[nt,:] = fmts.homogenisation(v, mesh, sigmaL, [0,1], sigmaEpsL).flatten() # zero for v
    
    sigmaL0, sigmaEpsL0 = fmts.getSigma_SigmaEps(param[0:2],mesh,np.zeros((2,2)))
    for j in range(3):
        X = scalerX[j].transform(ellipseData.reshape((1,nX)))
        # Y = models[j].predict(X)
        # alpha = scalerY[j].inverse_transform(Y)
        alpha = Y[j][nt,:].reshape((1,nY))
        
        for i in range(nY):
            vtemp = vtemp + x_epsL[j]*alpha[0,i]*interpolate(Wbasis_func[j][i],V).vector()[:]
    
    v.vector()[:] = vtemp
    
    if(op_solveForInternal):
        Ured = mpms.solveMultiscale(param[0:2,:], mesh, epsL, op = 'BCdirich_lag', others = [v])
        sigma_hat[nt,:] = fmts.homogenisation(Ured[0], mesh, sigmaL, [0,1], sigmaEpsL).flatten()
        Urec.append(Ured[0])
    else:    
        sigma_hat[nt,:] = sigma_bar[nt,:] + fmts.homogenisation(v, mesh, sigmaL0, [0,1], sigmaEpsL0).flatten()
        Urec.append(v)


print(sigma_hat)
print(sigma_bar)

nameStress = base_offline_folder + 'sigmaList{0}.txt'
sigma_true = np.loadtxt(nameStress.format(0))[:Ntest,:]

print(sigma_hat - sigma_true)

print(np.linalg.norm(sigma_hat - sigma_true,axis = 1))

# [[ 0.02566463 -0.00011596 -0.00011596  0.00309148]
#  [ 0.01889977 -0.00036228 -0.00036228  0.00156325]]
# [0.02585067 0.01897123]

# [[ 0.02537968 -0.00117215 -0.00117215  0.00241872]
#  [ 0.01827941 -0.00119904 -0.00119904  0.00133711]]
# [0.02554851 0.01840652]

# [[ 0.02540601 -0.00119975 -0.00119975  0.00251479]
#  [ 0.01830328 -0.00121823 -0.00121823  0.0014652 ]]
# [0.02558649 0.01844248]

# [[ 2.57050135e-02 -9.00922485e-05 -9.00922485e-05  3.24032885e-03]
#  [ 1.89795688e-02 -3.48659313e-04 -3.48659313e-04  1.71695801e-03]]
# [0.02590876 0.01906345]


# [[ 0.02493421 -0.00046445 -0.00046445  0.00033066]
#  [ 0.01799513 -0.00056488 -0.00056488 -0.00167002]]
# [0.03074962 0.0007313  0.0007313  0.00170244]

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


# Run_id = '1'
# EPOCHS = 500

# num_parameters = 0 # parameters that are taking into account into the model for minimization (split the norm)

# start = timer()

# 1)
# Neurons= 4*[32]
# drps = 6*[0.0]
# lr2 = 0.00001
# lr = 0.001
# decay = 0.1



# 2)
# Neurons= 4*[64]
# drps = 6*[0.0]
# lr2 = 0.00001
# lr = 0.001
# decay = 0.1



# 3)
# Neurons= 4*[64]
# drps = 6*[0.0]
# lr2 = 0.00001
# lr = 0.01
# decay = 0.0000001




# history = mytf.my_train_model( model, X, Y, num_parameters, EPOCHS, lr = lr, decay = decay, w_l = 1.0, w_mu = 0.0)
    # 
# mytf.plot_history( history, savefile = 'plot_history_0_6.png')

# with open(partialRadical + 'history_twoPoints_dispBound' + Run_id + '.dat', 'wb') as f:
#     pickle.dump(history.history, f)
    
# model.save_weights('weights_0_6')

# end = timer()
# print('time', end - start) 

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
    