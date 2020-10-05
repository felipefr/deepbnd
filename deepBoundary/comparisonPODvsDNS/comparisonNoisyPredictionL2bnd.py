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

dotProductL2bnd = lambda u,v, m : assemble(inner(u,v)*ds)
dotProductH10 = lambda u,v, dx : assemble(inner(grad(u),grad(v))*dx)
dotProductL2 = lambda u,v, dx : assemble(inner(u,v)*dx) 

simul_id = 3
EpsDirection = 0
DATAfolder = "/Users/felipefr/EPFL/newDLPDES/DATA/"
base_offline_folder = DATAfolder + "deepBoundary/data{0}/".format(simul_id)
folderDNS_meshes = DATAfolder + "deepBoundary/comparisonPODvsDNS/meshes/"
folderDNS_Per = DATAfolder + "deepBoundary/comparisonPODvsDNS/Per/"
folderDNS_Lin = DATAfolder + "deepBoundary/comparisonPODvsDNS/Lin/"
folderDNS_MR = DATAfolder + "deepBoundary/comparisonPODvsDNS/MR/"

nameMeshPrefix = folderDNS_meshes + "RVE_POD_reduced_{0}.{1}"

seedI = 0
seedE = 19
maxOffset = 2
ntest = seedE - seedI + 1

stressRad_Per = folderDNS_Per + 'sigmaL_{0}.hd5'
stressRad_Lin = folderDNS_Lin + 'sigmaL_{0}.hd5'
stressRad_MR = folderDNS_MR + 'sigmaL_{0}.hd5'

modelDNS_Per = 'periodic'
modelDNS_Lin = 'Lin'
modelDNS_MR = 'MR'

stressDNS_Per = myhd.loadhd5(stressRad_Per.format(modelDNS_Per),'SigmaL')[:,:,[0,3,1]] # voigt
stressDNS_Lin = myhd.loadhd5(stressRad_Lin.format(modelDNS_Lin),'SigmaL')[:,:,[0,3,1]] # voigt
stressDNS_MR = myhd.loadhd5(stressRad_MR.format(modelDNS_MR),'SigmaL')[:,:,[0,3,1]] # voigt

# nYlist = np.array([2,5,8,12,16,20,25,30,35,40]).astype('int')
nYlist = np.array([0,2,5,8,12,16,20,25,30,35,40,60,90,120,150,156]).astype('int')
Nmax = np.max(nYlist)

nameYlist = DATAfolder + 'deepBoundary/comparisonPODvsDNS/' + 'Y_{0}.hd5' 
nameTau = DATAfolder + 'deepBoundary/comparisonPODvsDNS/' + 'tau_{0}.hd5' 
# nameWlist =  DATAfolder + 'deepBoundary/training3Nets/definitiveBasis/Wbasis_{0}_3_0.hd5'

Ylist_L2bnd = myhd.loadhd5(nameYlist.format('L2bnd'),'Ylist')
tau_L2bnd, tau0_L2bnd, tau0fluc_L2bnd = myhd.loadhd5(nameTau.format('L2bnd'),['tau','tau_0','tau_0_fluc'])


# Wbasis_L2bnd = myhd.loadhd5(nameWlist.format('L2bnd_converted'),'Wbasis')     
stressPOD_L2bnd = np.zeros((len(nYlist),ntest,3)) 
stressPOD_L2bnd[0,:,:] = tau0_L2bnd[:ntest,:] + tau0fluc_L2bnd[:ntest,:] 

NpertLevels = 3
Npert = 100
pertLevels = [0.0001,0.001,0.01]

stressPOD_L2bnd_pert = np.zeros((NpertLevels,Npert,len(nYlist),ntest,3)) 

for i, nY in enumerate(nYlist[1:]):
    stressPOD_L2bnd[i+1,:,:] = np.einsum('ijk,ij->ik',tau_L2bnd[:ntest,:nY,:],Ylist_L2bnd[:ntest,:nY]) + tau0_L2bnd[:ntest,:] + tau0fluc_L2bnd[:ntest,:]

for k in range(NpertLevels):
    for j in range(Npert):
        stressPOD_L2bnd_pert[k,j,0,:,:] = tau0_L2bnd[:ntest,:] + tau0fluc_L2bnd[:ntest,:] 
        for i, nY in enumerate(nYlist[1:]):
            stressPOD_L2bnd_pert[k,j,i+1,:,:]  = np.einsum('ijk,ij->ik',tau_L2bnd[:ntest,:nY,:],(1.0 + pertLevels[k]*np.random.rand(ntest,nY))*Ylist_L2bnd[:ntest,:nY]) + tau0_L2bnd[:ntest,:] + tau0fluc_L2bnd[:ntest,:]      

# PLOOTTS 
rveSizes = 2 + 2*np.arange(3)
    


# referenceStress = (stressDNS_Lin[:,-1,0] + stressDNS_MR[:,-1,0] + stressDNS_Per[:,-1,0])/3.
referenceStress = stressDNS_Per[:,-1,0]
referenceStress_all = stressDNS_Per[:,-1,:]
normVoigt = lambda a : np.sqrt(a[0]**2.0 + a[1]**2.0 + 2.0*a[2]**2)

error_rel = lambda x, x0:  np.array([ [ (x[i,j] - x0[i])/x0[i] for j in range(len(x[0])) ] for i in range(len(x0))])


error_rel_all = lambda x, x0:  np.array([ [ normVoigt(x[i,j,:] - x0[i,:])/normVoigt(x0[i,:]) for j in range(len(x[0])) ] for i in range(len(x0))])


errorDNS_Per = error_rel( stressDNS_Per[:,:,0], referenceStress) 
errorDNS_MR = error_rel( stressDNS_MR[:,:,0], referenceStress)
errorDNS_Lin = error_rel( stressDNS_Lin[:,:,0], referenceStress)
errorPOD_L2bnd = error_rel( stressPOD_L2bnd[:,:,0].T, referenceStress) 
errorPOD_L2bnd_pert = []
errorPOD_L2bnd_pert.append( error_rel( np.mean(stressPOD_L2bnd_pert[0,:,:,:,0],axis = 0).T, referenceStress) )
errorPOD_L2bnd_pert.append( error_rel( np.mean(stressPOD_L2bnd_pert[1,:,:,:,0],axis = 0).T, referenceStress) )
errorPOD_L2bnd_pert.append( error_rel( np.mean(stressPOD_L2bnd_pert[2,:,:,:,0],axis = 0).T, referenceStress) )

errorDNS_Per_all = error_rel_all( stressDNS_Per, referenceStress_all) 
errorDNS_MR_all = error_rel_all( stressDNS_MR, referenceStress_all)
errorDNS_Lin_all = error_rel_all( stressDNS_Lin, referenceStress_all)
errorPOD_L2bnd_all = error_rel_all( np.transpose(stressPOD_L2bnd, axes = [1,0,2]) , referenceStress_all) 
errorPOD_L2bnd_pert_all = []
errorPOD_L2bnd_pert_all.append( error_rel_all( np.transpose(np.mean(stressPOD_L2bnd_pert[0,:,:,:,:],axis = 0), axes = [1,0,2]), referenceStress_all) )
errorPOD_L2bnd_pert_all.append( error_rel_all( np.transpose(np.mean(stressPOD_L2bnd_pert[1,:,:,:,:],axis = 0), axes = [1,0,2]), referenceStress_all) )
errorPOD_L2bnd_pert_all.append( error_rel_all( np.transpose(np.mean(stressPOD_L2bnd_pert[2,:,:,:,:],axis = 0), axes = [1,0,2]), referenceStress_all) ) 


fig = plt.figure(1, (6,4))
plt.title('POD error with noisy parameters')
plt.xlabel('N')
plt.yscale('log')
plt.ylabel(r'relative error stress for $(\cdot)_{11}$')
plt.ylim([1.0e-11,0.1])
plt.grid()
plut.plotMeanAndStd_noStdLegend(nYlist, np.abs(errorPOD_L2bnd), '$\\bar{\mathrm{e}}_N$', linetypes = ['k-o','k--','k--'], axis = 0)
for i, p in enumerate(pertLevels):
    lines = [ ['b', 'r', 'g'][i] + l for l in ['-o', '--', '--']  ]
    label = '$\\bar{\\bar{\mathrm{e}}}_{N,\delta = {%s}}$'%(['10^{-4}','10^{-3}','10^{-2}'][i])
    plut.plotMeanAndStd_noStdLegend(nYlist, np.abs(errorPOD_L2bnd_pert[i]) , label , lines, axis = 0)
plt.legend(loc = 8)
plt.savefig("error_log_noisy.pdf")


fig = plt.figure(2, (6,4))
plt.title('POD error with noisy parameters')
plt.xlabel('N')
plt.ylabel(r'relative error stress in Frobenius norm')
plt.ylim([1.0e-11,0.1])
plt.yscale('log')
plt.grid()
plut.plotMeanAndStd_noStdLegend(nYlist, errorPOD_L2bnd_all , '$\\bar{\mathrm{e}}_N$', linetypes = ['k-o','k--','k--'], axis = 0)
for i, p in enumerate(pertLevels):
    lines = [ ['b', 'r', 'g'][i] + l for l in ['-o', '--', '--']  ]
    label = '$\\bar{\\bar{\mathrm{e}}}_{N,\delta = {%s}}$'%(['10^{-4}','10^{-3}','10^{-2}'][i])
    plut.plotMeanAndStd_noStdLegend(nYlist, errorPOD_L2bnd_pert_all[i], label, lines, axis = 0) 
plt.legend(loc =8)
plt.savefig("error_log_allComponents_noisy.pdf")


e = np.zeros((4,len(nYlist)))

e[0,:] = np.mean(errorPOD_L2bnd_all, axis = 0)
for j in range(3):
    e[j+1,:] = np.mean(errorPOD_L2bnd_pert_all[j], axis = 0)

fig = plt.figure(3, (6,4))
plt.title('POD error with noisy parameters')
plt.xlabel('Noise Level')
plt.ylabel(r'relative error stress in Frobenius norm')
plt.ylim([1.0e-8,0.1])
plt.yscale('log')
plt.xscale('log')
plt.grid()
for i in [1,2,3,4,10,12,14]:
    plt.plot([1.0e-5] + pertLevels, e[:,i], '-o', label = 'N={%s}'%(str(nYlist[i])))
plt.xticks([1.0e-5] + pertLevels, ['POD', '$\delta = 10^{-4}$', '$\delta = 10^{-3}$', '$\delta = 10^{-2}$'])

plt.legend(loc =4)
plt.savefig("error_withNoiseLevel.pdf")





plt.show()
    
    