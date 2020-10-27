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

alpha = np.zeros(15)
beta = np.zeros(15)
A = np.zeros((20,15))

for i in range(20):
    for j in range(15):
        A[i,j] = np.abs(Ylist_L2bnd[i,j])/(np.max(Ylist_L2bnd[:,j]) - np.min(Ylist_L2bnd[:,j]))
        
for j in range(14):
    alpha[j] = np.min(A[:,:j+1])
    beta[j] = np.max(A[:,:j+1])

plt.figure(1)
plt.title("Coefficients Limits")
plt.plot(np.arange(14),alpha[:14],label = r'$\alpha$')
plt.plot(np.arange(14),beta[:14],label = r'$\beta$')
plt.legend()
plt.yscale('log')
plt.savefig('alpha-beta.pdf')
    plt.grid()
plt.show()
    
    