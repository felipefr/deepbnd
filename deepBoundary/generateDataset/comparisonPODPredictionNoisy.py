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
import plotUtils as plut
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
import myHDF5 as myhd
from basicTraining_lib import *

comm = MPI.comm_world


from basicTraining_lib import *

dotProduct = lambda u,v, ds : assemble(inner(u,v)*ds)

DATAfolder = "/Users/felipefr/EPFL/newDLPDES/DATA/"
base_offline_folder = DATAfolder + "deepBoundary/generateDataset/"
folderDNS = base_offline_folder + "axial/"
nameYlist = folderDNS + 'Y.h5' 
nameTau = folderDNS + 'tau2.h5' 
nameWlist =  folderDNS + 'Wbasis.h5'


maxOffset = 6
ntest = 13

nameSnapsDNS = folderDNS + 'snapshots.h5'

stressDNS = myhd.loadhd5(nameSnapsDNS,'sigma')[:ntest,:]
Ylist = myhd.loadhd5(nameYlist,'Ylist')[:ntest,:]
tau, tau0 = myhd.loadhd5(nameTau,['tau','tau_0'])
tau = tau[:ntest,:,:]
tau0 = tau0[:ntest,:]

nYlist = np.array([0,2,5,8,12,16,20,25,30,35,40,60,90,100,200,300,399]).astype('int')
Nmax = np.max(nYlist)

stressPOD = np.zeros((len(nYlist),ntest,3)) 
stressPOD[0,:,:] = tau0[:ntest,:]

NpertLevels = 3
Npert = 100
pertLevels = [0.0001,0.001,0.01]

stressPOD_pert = np.zeros((NpertLevels,Npert,len(nYlist),ntest,3)) 

for i, nY in enumerate(nYlist[1:]):
    stressPOD[i+1,:,:] = np.einsum('ijk,ij->ik',tau[:ntest,:nY,:],Ylist[:ntest,:nY]) + tau0[:ntest,:]

for k in range(NpertLevels):
    for j in range(Npert):
        stressPOD_pert[k,j,0,:,:] = tau0[:ntest,:]
        for i, nY in enumerate(nYlist[1:]):
            stressPOD_pert[k,j,i+1,:,:]  = np.einsum('ijk,ij->ik',tau[:ntest,:nY,:],(1.0 + pertLevels[k]*np.random.rand(ntest,nY))*Ylist[:ntest,:nY]) + tau0[:ntest,:]    

# PLOOTTS 
rveSizes = 2 + 2*np.arange(3)
    
referenceStress = stressDNS[:,0]
referenceStress_all = stressDNS[:,:]
normVoigt = lambda a : np.sqrt(a[0]**2.0 + a[1]**2.0 + 2.0*a[2]**2)

error_rel = lambda x, x0:  np.array([ [ (x[i,j] - x0[i])/x0[i] for j in range(len(x[0])) ] for i in range(len(x0))])
error_rel_all = lambda x, x0:  np.array([ [ normVoigt(x[i,j,:] - x0[i,:])/normVoigt(x0[i,:]) for j in range(len(x[0])) ] for i in range(len(x0))])

errorPOD = error_rel( stressPOD[:,:,0].T, referenceStress) 
errorPOD_pert = []
errorPOD_pert.append( error_rel( np.mean(stressPOD_pert[0,:,:,:,0],axis = 0).T, referenceStress) )
errorPOD_pert.append( error_rel( np.mean(stressPOD_pert[1,:,:,:,0],axis = 0).T, referenceStress) )
errorPOD_pert.append( error_rel( np.mean(stressPOD_pert[2,:,:,:,0],axis = 0).T, referenceStress) )

errorPOD_all = error_rel_all( np.transpose(stressPOD, axes = [1,0,2]) , referenceStress_all) 
errorPOD_pert_all = []
errorPOD_pert_all.append( error_rel_all( np.transpose(np.mean(stressPOD_pert[0,:,:,:,:],axis = 0), axes = [1,0,2]), referenceStress_all) )
errorPOD_pert_all.append( error_rel_all( np.transpose(np.mean(stressPOD_pert[1,:,:,:,:],axis = 0), axes = [1,0,2]), referenceStress_all) )
errorPOD_pert_all.append( error_rel_all( np.transpose(np.mean(stressPOD_pert[2,:,:,:,:],axis = 0), axes = [1,0,2]), referenceStress_all) ) 


fig = plt.figure(1, (6,4))
plt.title('POD error with noisy parameters')
plt.xlabel('N')
plt.ylabel(r'relative error stress for $(\cdot)_{11}$')
plt.ylim([1.0e-11,0.1])
plt.grid()
plut.plotMeanAndStd_noStdLegend(nYlist, np.abs(errorPOD), '$\\bar{\mathrm{e}}_N$', linetypes = ['k-o','k--','k--'], axis = 0)
# for i, p in enumerate(pertLevels):
#     lines = [ ['b', 'r', 'g'][i] + l for l in ['-o', '--', '--']  ]
#     label = '$\\bar{\\bar{\mathrm{e}}}_{N,\delta = {%s}}$'%(['10^{-4}','10^{-3}','10^{-2}'][i])
#     plut.plotMeanAndStd_noStdLegend(nYlist, np.abs(errorPOD_pert[i]) , label , lines, axis = 0)
plt.legend(loc = 'best')
plt.ylim(1.0e-5,1.0e-1)
plt.yscale('log')
plt.savefig("error_log_noisy.pdf")


fig = plt.figure(2, (6,4))
plt.title('POD error with noisy parameters')
plt.xlabel('N')
plt.ylabel(r'relative error stress in Frobenius norm')
# plt.ylim([1.0e-11,0.1])
plt.yscale('log')
plt.grid()
plut.plotMeanAndStd_noStdLegend(nYlist, errorPOD_all , '$\\bar{\mathrm{e}}_N$', linetypes = ['k-o','k--','k--'], axis = 0)
# for i, p in enumerate(pertLevels):
#     lines = [ ['b', 'r', 'g'][i] + l for l in ['-o', '--', '--']  ]
#     label = '$\\bar{\\bar{\mathrm{e}}}_{N,\delta = {%s}}$'%(['10^{-4}','10^{-3}','10^{-2}'][i])
#     plut.plotMeanAndStd_noStdLegend(nYlist, errorPOD_pert_all[i], label, lines, axis = 0) 
plt.legend(loc=8)
plt.savefig("error_log_allComponents_noisy.pdf")


# e = np.zeros((4,len(nYlist)))

# e[0,:] = np.mean(errorPOD_all, axis = 0)
# for j in range(3):
#     e[j+1,:] = np.mean(errorPOD_pert_all[j], axis = 0)

# fig = plt.figure(3, (6,4))
# plt.title('POD error with noisy parameters')
# plt.xlabel('Noise Level')
# plt.ylabel(r'relative error stress in Frobenius norm')
# plt.ylim([1.0e-8,0.1])
# plt.yscale('log')
# plt.xscale('log')
# plt.grid()
# for i in [1,2,3,4,10,12,14]:
#     plt.plot([1.0e-5] + pertLevels, e[:,i], '-o', label = 'N={%s}'%(str(nYlist[i])))
# plt.xticks([1.0e-5] + pertLevels, ['POD', '$\delta = 10^{-4}$', '$\delta = 10^{-3}$', '$\delta = 10^{-2}$'])

# plt.legend(loc =4)
# plt.savefig("error_withNoiseLevel.pdf")





plt.show()
    
    