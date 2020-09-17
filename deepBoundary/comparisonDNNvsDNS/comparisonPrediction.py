import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0,'../')
sys.path.insert(0,'../../utils/')
sys.path.insert(0,'../training3Nets/')
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
DATAfolder = "/Users/felipefr/EPFL/newDLPDES/DATA/"
base_offline_folder = DATAfolder + "deepBoundary/data{0}/".format(simul_id)
folderDNS = DATAfolder + "deepBoundary/comparisonDNNvsDNS/coarseP1/"


seedI = 0
seedE = 4
maxOffset = 8
ntest = seedE - seedI + 1
stressDNS = np.zeros((ntest,maxOffset  + 1 ,3))
stressRad = folderDNS + 'sigmaL_{0}_offset{1}_{2}.txt'
radiusRad = folderDNS + 'ellipseData_{0}.txt'

modelDNS = 'periodic'
radiusDNS = np.zeros((ntest,(2+2*maxOffset)**2))

for i in range(nt):
    radiusDNS[i,:] = np.loadtxt(radiusRad.format(i))[:,2]
    for j in range(maxOffset + 1 ):
        stressDNS[i,j,:] = np.loadtxt(stressRad.format(modelDNS,j,i))[[0,3,1]]

        
# fig , ax = plt.subplots()
# for i in range(nt):
#     plt.plot(stressDNS[i,:,0])
# plt.xlabel('size')

# plt.grid()
# Nlist = [5*i for i in range(8)]
# ax1 = ax.twinx()
# # ax1.plot(Nlist , np.mean(stressDNS[:,:,0],axis=0),'--')
# # ax1.plot(Nlist , np.mean(stressDNS[:,:,0],axis=0) + np.std(stressDNS[:,:,0],axis=0),'--')
# # ax1.plot(Nlist , np.mean(stressDNS[:,:,0],axis=0) - np.std(stressDNS[:,:,0],axis=0),'--')
# for i in range(nt):
#     ax1.plot(np.abs(stressDNS[i,:-1,0] - stressDNS[i,-1,0])/stressDNS[i,-1,0],'--')

# # ax1.plot(np.mean(np.abs(stressDNS[:,:-1,0] - stressDNS[:,-1,0])/stressDNS[:,-1,0]),'k', '..')
# ax1.set_ylabel('Error')
# ax1.set_yscale('log')
# plt.show()

arch_id = 1

nX = 4 # because the only first 4 are relevant, the other are constant

print("starting with", simul_id, EpsDirection, arch_id)


# Arch 1
net = {'Neurons': 5*[128], 'drps': 7*[0.0], 'activations': ['relu','relu','relu'], 
        'reg': 0.00001, 'lr': 0.01, 'decay' : 0.5, 'epochs': 1000}

## Arch 2
## net = {'Neurons': [128,256,256,256,512,1024], 'drps': 8*[0.1], 'activations': ['relu','relu','relu'], 
##         'reg': 0.0001, 'lr': 0.001, 'decay' : 0.5, 'epochs': 1000}


fnames = {}      

## ### fnames['prefix_out'] = 'saves_PODMSE_comparisonLessNy_Eps{0}_arch{1}/'.format(EpsDirection, arch_id)
## fnames['prefix_out'] = 'saves_correct_PODMSE_L2bnd_Eps{0}_arch{1}/'.format(EpsDirection, arch_id)
fnames['prefix_out'] = DATAfolder + 'deepBoundary/training3Nets/saves_correct_PODMSE_H10_Eps{0}_arch{1}/'.format(EpsDirection, arch_id)
fnames['prefix_in_X'] = base_offline_folder + "ellipseData_{0}.txt"
## fnames['prefix_in_Y'] = "./definitiveBasis/Y_L2bnd_original_{0}_{1}.hd5".format(simul_id, EpsDirection)
fnames['prefix_in_Y'] = DATAfolder + 'deepBoundary/training3Nets/definitiveBasis/Y_H10_lite2_correction_{0}_{1}.hd5'.format(simul_id, EpsDirection)

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
  

nx = 100
meshRef = RectangleMesh(Point(1.0/3., 1./3.), Point(2./3., 2./3.), nx, nx, diagonal='crossed')
Vref = VectorFunctionSpace(meshRef,"CG", 2)

dxRef = Measure('dx', meshRef) 
dsRef = Measure('ds', meshRef) 

# ## nameInterpolation = 'interpolatedSolutions_{0}_{1}.txt'
nameInterpolation = DATAfolder + 'deepBoundary/training3Nets/definitiveBasis/interpolatedSolutions_lite2_{0}_{1}.hd5'
Isol = myloadfile(nameInterpolation.format(simul_id,EpsDirection),'Isol')

nameWbasis = 'deepBoundary/training3Nets/definitiveBasis/Wbasis_{0}{1}_{2}.hd5'
nameYlist = 'deepBoundary/training3Nets/definitiveBasis/Y_{0}{1}_{2}.hd5' ### need to generate
nameStress = base_offline_folder + 'sigmaList{0}.txt'
nameTau = 'deepBoundary/training3Nets/definitiveBasis/tau_H10_lite2_correction_3_0.hd5'  ### need to generate
## nameTau = './definitiveBasis/tau_L2bnd_original_solvePDE_complete_3_0.hd5'

Nmax = np.max(nYlist) # 150 
Wbasis = myloadfile(nameWbasis.format('H10_lite2_correction_', simul_id,EpsDirection),'Wbasis')[:Nmax,:]
Ylist = myloadfile(nameYlist.format('H10_lite2_correction_', simul_id,EpsDirection),'Ylist')

tau, tau0 = myhd.loadhd5(nameTau,['tau','tau_0'])
sigmaList = np.loadtxt(nameStress.format(EpsDirection))[:ntest,[0,3,1]] # flatten to voigt notation
        
# mseErrors = {}
# mseErrors['POD_norm_stress'] = gdb.getMSEstresses(nYlist,Ylist[:ntest,:],tau,tau0,sigmaList)
# mseErrors['POD_norm_L2bnd'] = gdb.getMSE(nYlist,Ylist[:ntest,:],Wbasis, Isol[:ntest,:], Vref, dsRef, dotProductL2bnd)
# mseErrors['POD_norm_H10'] = gdb.getMSE(nYlist,Ylist[:ntest,:],Wbasis, Isol[:ntest,:], Vref, dxRef, dotProductH10)
# mseErrors['POD_norm_L2'] = gdb.getMSE(nYlist,Ylist[:ntest,:],Wbasis, Isol[:ntest,:], Vref, dxRef, dotProductL2)


# errors = {}
# errors['L2bnd'] = []
# errors['H10'] = []
# errors['L2'] = []
# errors['stress'] = []

# run = 0
# for i, nY in enumerate(nYlist):
#     Yp_bar = models[i][run].predict(X[0][:ntest])
#     Yp = scalerY[0].inverse_transform(np.concatenate((Yp_bar, np.zeros((ntest,Nmax-nY))),axis = 1))[:,:nY]
#     errors['stress'].append(gdb.getMSEstresses([nY],Yp,tau,tau0,sigmaList)[0])
#     errors['L2bnd'].append(gdb.getMSE([nY],Yp,Wbasis, Isol[:ntest,:], Vref, dsRef, dotProductL2bnd)[0])
#     errors['H10'].append(gdb.getMSE([nY],Yp,Wbasis, Isol[:ntest,:], Vref, dxRef, dotProductH10)[0])
#     errors['L2'].append(gdb.getMSE([nY],Yp,Wbasis, Isol[:ntest,:], Vref, dxRef, dotProductL2)[0])
    
    
# mseErrors['model_norm_L2bnd'] = np.array(errors['L2bnd']) 
# mseErrors['model_norm_H10'] = np.array(errors['H10']) 
# mseErrors['model_norm_L2'] = np.array(errors['L2']) 
# mseErrors['model_norm_stress'] = np.array(errors['stress']) 


# plt.figure(3, (9,7))
# plt.suptitle("Average errors 100 snapshots (H10-based RB), Arch 1, Seed {0}".format(run+1))
# plt.subplot('221')
# plt.title('$L^2(\partial \Omega_{\mu})$')
# plt.plot(nYlist,mseErrors['POD_norm_L2bnd'], '-o')
# plt.plot(nYlist,mseErrors['model_norm_L2bnd'], '-o')
# plt.xlabel('N')
# plt.ylabel('error sqrt(mse)')
# plt.yscale('log')
# plt.grid()
# plt.legend(['POD', 'DNN'])
# plt.subplot('222')
# plt.title('$L^2(\Omega_{\mu})$')
# plt.plot(nYlist,mseErrors['POD_norm_L2'], '-o')
# plt.plot(nYlist,mseErrors['model_norm_L2'], '-o')
# plt.xlabel('N')
# plt.ylabel('error sqrt(mse)')
# plt.yscale('log')
# plt.grid()
# plt.legend(['POD', 'DNN'])
# plt.subplot('223')
# plt.title('$H^1_0(\Omega_{\mu})$')
# plt.plot(nYlist,mseErrors['POD_norm_H10'], '-o')
# plt.plot(nYlist,mseErrors['model_norm_H10'], '-o')
# plt.xlabel('N')
# plt.ylabel('error sqrt(mse)')
# plt.yscale('log')
# plt.grid()
# plt.legend(['POD', 'DNN'])
# plt.subplot('224')
# plt.title('Homogenised Stress')
# plt.plot(nYlist,mseErrors['POD_norm_stress'], '-o')
# plt.plot(nYlist,mseErrors['model_norm_stress'], '-o')
# plt.xlabel('N')
# plt.ylabel('error sqrt(mse)')
# plt.yscale('log')
# plt.grid()
# plt.legend(['POD', 'DNN']) 
# plt.tight_layout(rect=[0, 0.03, 1, 0.97])


# plt.savefig("comparisonPODMSE/norms_POD_DNN_H10_corrected_arch1_seed{0}.png".format(run+1))
# plt.show()

