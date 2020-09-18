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
folderDNS_Per = DATAfolder + "deepBoundary/comparisonPODvsDNS/Per/"
folderDNS_Lin = DATAfolder + "deepBoundary/comparisonPODvsDNS/Lin/"
folderDNS_MR = DATAfolder + "deepBoundary/comparisonPODvsDNS/MR/"

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



stressDNS_Per = myhd.loadhd5(stressRad_Per.format(modelDNS_Per),'SigmaL')
stressDNS_Lin = myhd.loadhd5(stressRad_Lin.format(modelDNS_Lin),'SigmaL')
stressDNS_MR = myhd.loadhd5(stressRad_MR.format(modelDNS_MR),'SigmaL')

nsTrain = 1000

nYlist = np.array([2,5,8,12,16,20,25,30,35,40]).astype('int')

nameYlist = 'Y_{0}.hd5' 
nameTau = 'tau_{0}.hd5' 

Ylist_H10 = myhd.loadhd5(nameYlist.format('H10'),'Ylist')
tau_H10, tau0_H10 = myhd.loadhd5(nameTau.format('H10'),['tau','tau_0'])

Ylist_L2bnd = myhd.loadhd5(nameYlist.format('H10'),'Ylist')
tau_L2bnd, tau0_L2bnd = myhd.loadhd5(nameTau.format('L2bnd'),['tau','tau_0'])
      
Nmax = np.max(nYlist)
stressPOD_H10 = np.zeros((len(nYlist)+1,ntest,3)) # zero case tested
stressPOD_H10[0,:,:] = tau0_H10[:ntest]

stressPOD_L2bnd = np.zeros((len(nYlist)+1,ntest,3)) # zero case tested
stressPOD_L2bnd[0,:,:] = tau0_L2bnd[:ntest]

for i, nY in enumerate(nYlist):
    stressPOD_H10[i+1,:,:] = np.einsum('ijk,ij->ik',tau_H10[:ntest,:nY,:],Ylist_H10[:ntest,:nY]) + tau0_H10[:ntest]
    stressPOD_L2bnd[i+1,:,:] = np.einsum('ijk,ij->ik',tau_L2bnd[:ntest,:nY,:],Ylist_L2bnd[:ntest,:nY]) + tau0_L2bnd[:ntest]


PLOOTTS 
for i in range(ntest):
    fig , ax = plt.subplots()
    plt.plot(2 + 2*np.arange(3), stressDNS_Per[i,:,0], '-o', label = 'periodic')
    plt.plot(2 + 2*np.arange(3), stressDNS_Lin[i,:,0], '-o', label = 'linear bnd')
    plt.plot(2 + 2*np.arange(3), stressDNS_MR[i,:,0], '-o', label = 'Minim. Rest.')
    plt.xlabel('size offset')
    plt.legend(loc = 1)
    plt.grid()
    # ax1 = ax.twiny()
    # plt.plot([0] + list(nYlist), stressPOD_H10[:,i,0], '--', label = 'POD H10' )
    # plt.plot([0] + list(nYlist), stressPOD_L2bnd[:,i,0], '--', label = 'POD L2bnd' )
    
    # ax1.set_xlabel('N RB')
    
    # plt.legend(loc = 2)
    plt.savefig("comparison_POD_test{0}.png".format(i))
    plt.show()
    
# fig = plt.figure((10,10))
fig , ax = plt.subplots()
plt.plot(2 + 2*np.arange(3), np.mean(stressDNS_Per[:,:,0],axis = 0), 'b-o', label = 'periodic')
plt.plot(2 + 2*np.arange(3), np.mean(stressDNS_Per[:,:,0],axis = 0) + np.std(stressDNS_Per[:,:,0],axis = 0) , 'b--', label = 'periodic + std')
plt.plot(2 + 2*np.arange(3), np.mean(stressDNS_Per[:,:,0],axis = 0) - np.std(stressDNS_Per[:,:,0],axis = 0) , 'b--', label = 'periodic - std')
plt.plot(2 + 2*np.arange(3), np.mean(stressDNS_Lin[:,:,0],axis = 0), 'r-o', label = 'linear')
plt.plot(2 + 2*np.arange(3), np.mean(stressDNS_Lin[:,:,0],axis = 0) + np.std(stressDNS_Lin[:,:,0],axis = 0) , 'r--', label = 'linear + std')
plt.plot(2 + 2*np.arange(3), np.mean(stressDNS_Lin[:,:,0],axis = 0) - np.std(stressDNS_Lin[:,:,0],axis = 0) , 'r--', label = 'linear - std')
plt.plot(2 + 2*np.arange(3), np.mean(stressDNS_MR[:,:,0],axis = 0), 'k-o', label = 'MR')
plt.plot(2 + 2*np.arange(3), np.mean(stressDNS_MR[:,:,0],axis = 0) + np.std(stressDNS_MR[:,:,0],axis = 0) , 'k--', label = 'MR + std')
plt.plot(2 + 2*np.arange(3), np.mean(stressDNS_MR[:,:,0],axis = 0) - np.std(stressDNS_MR[:,:,0],axis = 0) , 'k--', label = 'MR - std')
plt.xlabel('size RVE')
plt.ylabel('stress_hom_11')
plt.legend(loc = 1)
plt.grid()
ax1 = ax.twiny()
# plt.plot([0] + list(nYlist), np.mean(stressPOD_H10[:,:,0],axis = 1), '--', label = 'POD H10' )
plt.plot([0] + list(nYlist), np.mean(stressPOD_L2bnd[:,:,0],axis = 1), '-o', label = 'POD L2bnd' )
plt.plot([0] + list(nYlist), np.mean(stressPOD_L2bnd[:,:,0],axis = 1) + np.std(stressPOD_L2bnd[:,:,0],axis = 1), 'g--', label = 'POD L2bnd + std' )
plt.plot([0] + list(nYlist), np.mean(stressPOD_L2bnd[:,:,0],axis = 1) - np.std(stressPOD_L2bnd[:,:,0],axis = 1), 'g--', label = 'POD L2bnd - std' )
# plt.plot([0] + list(nYlist), stressPOD_L2bnd[:,i,0], '--', label = 'POD L2bnd' )

ax1.set_xlabel('N RB')

plt.legend(loc = 2)
plt.savefig("comparison_POD_withoutH10_avg.png")
plt.show()
    
    