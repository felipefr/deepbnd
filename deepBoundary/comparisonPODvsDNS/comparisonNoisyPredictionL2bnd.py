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

nameYlist = 'Y_{0}.hd5' 
nameTau = 'tau_{0}.hd5' 
# nameWlist =  DATAfolder + 'deepBoundary/training3Nets/definitiveBasis/Wbasis_{0}_3_0.hd5'

Ylist_H10 = myhd.loadhd5(nameYlist.format('H10'),'Ylist')
tau_H10, tau0_H10 , tau0fluc_H10 = myhd.loadhd5(nameTau.format('H10'),['tau','tau_0', 'tau_0_fluc'])

Ylist_L2bnd = myhd.loadhd5(nameYlist.format('L2bnd'),'Ylist')
tau_L2bnd, tau0_L2bnd, tau0fluc_L2bnd = myhd.loadhd5(nameTau.format('L2bnd'),['tau','tau_0','tau_0_fluc'])

Ylist_L2bnd_per = myhd.loadhd5(nameYlist.format('L2bnd'),'Ylist')
tau_L2bnd_per, tau0_L2bnd_per, tau0fluc_L2bnd_per = myhd.loadhd5(nameTau.format('L2bnd_per'),['tau','tau_0','tau_0_fluc'])

# Ylist_L2bnd_noOrth = myhd.loadhd5(nameYlist.format('L2bnd'),'Ylist')
# tau_L2bnd_noOrth, tau0_L2bnd_noOrth, tau0fluc_L2bnd_noOrth = myhd.loadhd5(nameTau.format('L2bnd_noOrth'),['tau','tau_0','tau_0_fluc'])

Ylist_L2bnd_noOrth = myhd.loadhd5(nameYlist.format('H10'),'Ylist')
tau_L2bnd_noOrth, tau0_L2bnd_noOrth, tau0fluc_L2bnd_noOrth = myhd.loadhd5(nameTau.format('H10_Orth_VL'),['tau','tau_0','tau_0_fluc'])

# tau_H10_Orth_VL.hd5 revenir
# tau_H10_noOrth_VL.hd5 really bad
#tau_H10_recomputed.hd5 validated 
# tau_H10_recomputed_Orth_VT.hd5 slightly worse than noOrth_VT (given by H10_recomputed). Reason , reinterpolation to the coarser mesh
# tau_L2bnd_noOrth.hd5 # really bad

# Wbasis_L2bnd = myhd.loadhd5(nameWlist.format('L2bnd_converted'),'Wbasis')

      

stressPOD_H10 = np.zeros((len(nYlist),ntest,3)) 
stressPOD_H10[0,:,:] = tau0_H10[:ntest] + tau0fluc_H10[:ntest,:] 

stressPOD_L2bnd = np.zeros((len(nYlist),ntest,3)) 
stressPOD_L2bnd[0,:,:] = tau0_L2bnd[:ntest,:] + tau0fluc_L2bnd[:ntest,:] 

stressPOD_L2bnd_per = np.zeros((len(nYlist),ntest,3)) 
stressPOD_L2bnd_per[0,:,:] = tau0_L2bnd_per[:ntest,:] + tau0fluc_L2bnd_per[:ntest,:] 

stressPOD_L2bnd_noOrth = np.zeros((len(nYlist),ntest,3)) 
stressPOD_L2bnd_noOrth[0,:,:] = tau0_L2bnd_noOrth[:ntest,:] + tau0fluc_L2bnd_noOrth[:ntest,:] 

for i, nY in enumerate(nYlist[1:]):
    stressPOD_H10[i+1,:,:] = np.einsum('ijk,ij->ik',tau_H10[:ntest,:nY,:],Ylist_H10[:ntest,:nY]) + tau0_H10[:ntest] + tau0fluc_H10[:ntest]
    stressPOD_L2bnd[i+1,:,:] = np.einsum('ijk,ij->ik',tau_L2bnd[:ntest,:nY,:],Ylist_L2bnd[:ntest,:nY]) + tau0_L2bnd[:ntest,:] + tau0fluc_L2bnd[:ntest,:]
    stressPOD_L2bnd_per[i+1,:,:] = np.einsum('ijk,ij->ik',tau_L2bnd_per[:ntest,:nY,:],Ylist_L2bnd_per[:ntest,:nY]) + tau0_L2bnd_per[:ntest,:] + tau0fluc_L2bnd_per[:ntest,:]
    stressPOD_L2bnd_noOrth[i+1,:,:] = np.einsum('ijk,ij->ik',tau_L2bnd_noOrth[:ntest,:nY,:],Ylist_L2bnd_noOrth[:ntest,:nY]) + tau0_L2bnd_noOrth[:ntest,:] + tau0fluc_L2bnd_noOrth[:ntest,:]



# PLOOTTS 
rveSizes = 2 + 2*np.arange(3)


    
# for i in range(ntest):
#     fig = plt.figure(i + 3, (7,5))
#     ax = fig.subplots()
#     plt.plot(rveSizes, stressDNS_Per[i,:,0], '-o', label = 'periodic')
#     plt.plot(rveSizes, stressDNS_Lin[i,:,0], '-o', label = 'linear bnd')
#     plt.plot(rveSizes, stressDNS_MR[i,:,0], '-o', label = 'Minim. Rest.')

#     plt.xlabel('size RVE')
#     plt.legend(loc = 1)
#     plt.grid()
#     ax1 = ax.twiny()
#     plt.plot(nYlist, stressPOD_L2bnd[:,i,0], '--o', label = 'POD L2bnd' )
#     ax1.set_xlabel('N RB')
#     plt.legend(loc = 2)
#     plt.savefig("stress_test{0}.png".format(i))
#     plt.show()
    
fig = plt.figure(1, (9,8))
ax = fig.subplots()
plut.plotMeanAndStd(rveSizes, stressDNS_Per[:,:,0] , 0, 'periodic' , ['b-o', 'b--', 'b--'])
plut.plotMeanAndStd(rveSizes, stressDNS_Lin[:,:,0] , 0, 'linear' , ['r-o', 'r--', 'r--'])
plut.plotMeanAndStd(rveSizes, stressDNS_MR[:,:,0] , 0, 'MR' , ['k-o', 'k--', 'k--'])
plt.xlabel('size RVE')
plt.ylabel('stress_hom_11')
plt.legend(loc = 1)
plt.grid()
ax1 = ax.twiny()
plut.plotMeanAndStd(nYlist, stressPOD_L2bnd[:,:,0], 1, 'POD L2bnd' , ['g-o', 'g--', 'g--'])
# plut.plotMeanAndStd(nYlist[3:], stressPOD_H10[3:,:,0], 1, 'POD H10' , ['m-o', 'm--', 'm--'])
# plut.plotMeanAndStd(nYlist, stressPOD_L2bnd_per[:,:,0], 1, 'POD L2bnd Per' , ['c-o', 'c--', 'c--'])
plut.plotMeanAndStd(nYlist[:], stressPOD_L2bnd_noOrth[:,:,0], 1, 'POD L2bnd noOrth' , ['m-o', 'm--', 'm--'])
ax1.set_xlabel('N RB')

plt.legend(loc = 2)
# plt.savefig("stress_avg_withPer.png")



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
errorPOD_H10 = error_rel( stressPOD_H10[:,:,0].T, referenceStress) 
errorPOD_L2bnd_per = error_rel( stressPOD_L2bnd_per[:,:,0].T, referenceStress) 
errorPOD_L2bnd_noOrth = error_rel( stressPOD_L2bnd_noOrth[:,:,0].T, referenceStress) 


errorDNS_Per_all = error_rel_all( stressDNS_Per, referenceStress_all) 
errorDNS_MR_all = error_rel_all( stressDNS_MR, referenceStress_all)
errorDNS_Lin_all = error_rel_all( stressDNS_Lin, referenceStress_all)
errorPOD_L2bnd_all = error_rel_all( np.transpose(stressPOD_L2bnd, axes = [1,0,2]) , referenceStress_all) 
errorPOD_H10_all = error_rel_all( np.transpose(stressPOD_H10, axes = [1,0,2]), referenceStress_all) 
errorPOD_L2bnd_per_all = error_rel_all( np.transpose(stressPOD_L2bnd_per, axes = [1,0,2]) , referenceStress_all) 
errorPOD_L2bnd_noOrth_all = error_rel_all( np.transpose(stressPOD_L2bnd_noOrth, axes = [1,0,2]) , referenceStress_all) 


fig = plt.figure(2, (9,8))
ax = fig.subplots()
plut.plotMeanAndStd(rveSizes, errorDNS_Per , 0, 'periodic', linetypes = ['b-o','b--','b--'])
plut.plotMeanAndStd(rveSizes, errorDNS_Lin , 0, 'linear', linetypes = ['r-o','r--','r--'])
plut.plotMeanAndStd(rveSizes, errorDNS_MR , 0, 'MR', linetypes = ['k-o','k--','k--'])
plt.xlabel('size RVE')
plt.ylabel('rel error stress')
plt.legend(loc = 1)
plt.grid()
ax1 = ax.twiny()
plut.plotMeanAndStd(nYlist , errorPOD_L2bnd[:,:], 0, 'POD L2bnd', linetypes = ['g-o','g--','g--'])
# plut.plotMeanAndStd(nYlist[3:], errorPOD_H10[:,3:], 0, 'POD H10', linetypes = ['m-o','m--','m--'])
# plut.plotMeanAndStd(nYlist, errorPOD_L2bnd_per[:,:], 0, 'POD L2bnd Per', linetypes = ['c-o','c--','c--'])
plut.plotMeanAndStd(nYlist, errorPOD_L2bnd_noOrth[:,:], 0, 'POD L2bnd noOrth', linetypes = ['m-o','m--','m--'])
ax1.set_xlabel('N RB')

plt.legend(loc = 2)
# plt.savefig("error_from5.png")

fig = plt.figure(3, (9,8))
ax = fig.subplots()
plut.plotMeanAndStd(rveSizes, np.abs(errorDNS_Per) , 0, 'periodic', linetypes = ['b-o','b--','b--'])
plut.plotMeanAndStd(rveSizes, np.abs(errorDNS_Lin) , 0, 'linear', linetypes = ['r-o','r--','r--'])
plut.plotMeanAndStd(rveSizes, np.abs(errorDNS_MR) , 0, 'MR', linetypes = ['k-o','k--','k--'])
plt.xlabel('size RVE')
plt.ylabel('rel error stress')
plt.ylim([1.0e-11,2.0])
plt.legend(loc = 3)
# plt.yscale('log')
plt.grid()
ax1 = ax.twiny()
plut.plotMeanAndStd(nYlist, np.abs(errorPOD_L2bnd), 0, 'POD L2bnd', linetypes = ['g-o','g--','g--'])
# plut.plotMeanAndStd(nYlist, np.abs(errorPOD_H10), 0, 'POD H10', linetypes = ['m-o','m--','m--'])
# plut.plotMeanAndStd(nYlist, np.abs(errorPOD_L2bnd_per), 0, 'POD L2bnd Per', linetypes = ['c-o','c--','c--'])
plut.plotMeanAndStd(nYlist, np.abs(errorPOD_L2bnd_noOrth), 0, 'POD L2bnd noOrth', linetypes = ['m-o','m--','m--'])
ax1.set_xlabel('N RB')
ax1.set_yscale('log')

plt.legend(loc ='best')
# plt.savefig("error_log.png")


fig = plt.figure(4, (9,8))
ax = fig.subplots()
plut.plotMeanAndStd(rveSizes, errorDNS_Per_all , 0, 'periodic', linetypes = ['b-o','b--','b--'])
plut.plotMeanAndStd(rveSizes, errorDNS_Lin_all , 0, 'linear', linetypes = ['r-o','r--','r--'])
plut.plotMeanAndStd(rveSizes, errorDNS_MR_all , 0, 'MR', linetypes = ['k-o','k--','k--'])
plt.xlabel('size RVE')
plt.ylabel('rel error stress (all components, Frobenius)')
plt.ylim([1.0e-11,2.0])
plt.legend(loc = 3)
# plt.yscale('log')
plt.grid()
ax1 = ax.twiny()
plut.plotMeanAndStd(nYlist, errorPOD_L2bnd_all , 0, 'POD L2bnd', linetypes = ['g-o','g--','g--'])
# plut.plotMeanAndStd(nYlist, errorPOD_H10_all, 0, 'POD H10', linetypes = ['m-o','m--','m--'])
# plut.plotMeanAndStd(nYlist, errorPOD_L2bnd_per_all , 0, 'POD L2bnd Per', linetypes = ['c-o','c--','c--'])
plut.plotMeanAndStd(nYlist, errorPOD_L2bnd_noOrth_all , 0, 'POD H10 Orth', linetypes = ['m-o','m--','m--'])
ax1.set_xlabel('N RB')
ax1.set_yscale('log')

plt.legend(loc =8)
# plt.savefig("error_log_allComponents.png")





plt.show()
    
    