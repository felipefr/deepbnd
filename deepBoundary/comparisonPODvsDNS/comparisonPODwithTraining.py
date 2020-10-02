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

DATAfolder = "/Users/felipefr/EPFL/newDLPDES/DATA/deepBoundary/data3/"
nameMeshPrefix = DATAfolder + "RVE_POD_reduced_{0}.{1}"

ntest = 20

stressRad = DATAfolder + 'sigmaList0.txt'

stressDNS = myhd.genericLoadfile(stressRad,'SigmaL')[:ntest,:]

nYlist = np.array([2,5,8,12,16,20,25,30,35,40,60,90,120,150,156]).astype('int')

nameYlist = 'Y_{0}.hd5' 
nameTau = 'tau_{0}.hd5' 
nameWlist =  "/Users/felipefr/EPFL/newDLPDES/DATA/deepBoundary/training3Nets/definitiveBasis/Wbasis_{0}_3_0.hd5"

Ylist_L2bnd = myhd.loadhd5(nameYlist.format('train_L2bnd'),'Ylist')
tau_L2bnd, tau0_L2bnd, tau0fluc_L2bnd = myhd.loadhd5(nameTau.format('train_L2bnd'),['tau','tau_0','tau_0_fluc'])

Wbasis_L2bnd = myhd.loadhd5(nameWlist.format('L2bnd_converted'),'Wbasis')

Nmax = np.max(nYlist)

stressPOD_L2bnd = np.zeros((len(nYlist)+1,ntest,3)) # zero case tested
stressPOD_L2bnd[0,:,:] = tau0_L2bnd[:ntest,:] + tau0fluc_L2bnd[:ntest,:] 

for i, nY in enumerate(nYlist):
    stressPOD_L2bnd[i+1,:,:] = np.einsum('ijk,ij->ik',tau_L2bnd[:ntest,:nY,:],Ylist_L2bnd[:ntest,:nY]) + tau0_L2bnd[:ntest,:] + tau0fluc_L2bnd[:ntest,:]


# PLOOTTS 
for i in range(ntest):
    fig = plt.figure(i + 3, (7,5))
    plt.plot([0,nYlist[-1]] , 2*[stressDNS[i,0]], '-o', label = 'DNS')
    plt.plot([0] + list(nYlist), stressPOD_L2bnd[:,i,0], '--o', label = 'POD L2bnd' )
    plt.xlabel('N RB')
    plt.ylabel('stress')
    plt.legend(loc = 1)
    plt.grid()

    plt.savefig("comparison_POD_test_corrected_train{0}.png".format(i))
    plt.show()
    
fig = plt.figure(1, (9,8))
plt.plot([0,nYlist[-1]], 2*[np.mean(stressDNS[:,0],axis = 0)], 'b-o', label = 'DNS')
plt.plot([0,nYlist[-1]], 2*[np.mean(stressDNS[:,0],axis = 0) + np.std(stressDNS[:,0],axis = 0)], 'b--', label = 'DNS + str')
plt.plot([0,nYlist[-1]], 2*[np.mean(stressDNS[:,0],axis = 0) - np.std(stressDNS[:,0],axis = 0)], 'b--', label = 'DNS - str')
plt.plot([0] + list(nYlist), np.mean(stressPOD_L2bnd[:,:,0],axis = 1), 'g-o', label = 'POD L2bnd' )
plt.plot([0] + list(nYlist), np.mean(stressPOD_L2bnd[:,:,0],axis = 1) + np.std(stressPOD_L2bnd[:,:,0],axis = 1), 'g--', label = 'POD L2bnd + std' )
plt.plot([0] + list(nYlist), np.mean(stressPOD_L2bnd[:,:,0],axis = 1) - np.std(stressPOD_L2bnd[:,:,0],axis = 1), 'g--', label = 'POD L2bnd - std' )
plt.ylabel('stress_hom_11')
plt.xlabel('N RB')
plt.legend(loc = 2)
plt.grid()
plt.savefig("comparison_POD_corrected_avg_train.png")



referenceStress = stressDNS[:,0]
error_rel = lambda x, x0:  np.array([ [ (x[i,j] - x0[i])/x0[i] for j in range(len(x[0])) ] for i in range(len(x0))])

errorPOD_L2bnd = error_rel( stressPOD_L2bnd[:,:,0].T, referenceStress) 


fig = plt.figure(2, (9,8))
plt.plot([0] + list(nYlist), np.mean(errorPOD_L2bnd,axis = 0), 'g-o', label = 'POD L2bnd' )
plt.plot([0] + list(nYlist), np.mean(errorPOD_L2bnd,axis = 0) + np.std(errorPOD_L2bnd,axis = 0), 'g--', label = 'POD L2bnd + std' )
plt.plot([0] + list(nYlist), np.mean(errorPOD_L2bnd,axis = 0) - np.std(errorPOD_L2bnd,axis = 0), 'g--', label = 'POD L2bnd - std' )
plt.ylabel('rel error stress')
plt.xlabel('N RB')
plt.legend(loc = 2)
plt.grid()
plt.savefig("comparison_error_POD_corrected_train.png")



fig = plt.figure(3, (9,8))
plt.plot([0] + list(nYlist), np.abs(np.mean(errorPOD_L2bnd,axis = 0)), 'g-o', label = 'POD L2bnd' )
plt.ylabel('rel error stress')
plt.xlabel('N RB')
plt.legend(loc = 2)
plt.yscale('log')
plt.grid()
plt.savefig("comparison_error_POD_corrected_train_log.png")




plt.show()
    
    