import sys, os
from numpy import isclose
from dolfin import *
# import matplotlib.pyplot as plt
from multiphenics import *
sys.path.insert(0, '../../utils/')

import fenicsWrapperElasticity as fela
import matplotlib.pyplot as plt
import numpy as np
import generatorMultiscale as gmts
import wrapperPygmsh as gmsh
import generationInclusions as geni
import myCoeffClass as coef
import fenicsMultiscale as fmts
import elasticity_utils as elut
import fenicsWrapperElasticity as fela
import multiphenicsMultiscale as mpms
import ioFenicsWrappers as iofe
import fenicsUtils as feut

import matplotlib.pyplot as plt
import copy
import myHDF5 as myhd

folder = ["/Users", "/home"][0] + "/felipefr/EPFL/newDLPDEs/DATA/deepBoundary/comparisonPODvsDNS/Lin/"
folderMesh = ["/Users", "/home"][0] + "/felipefr/EPFL/newDLPDEs/DATA/deepBoundary/comparisonPODvsDNS/meshes/"

radFile = folder + "RVE_POD_{0}.{1}"
radFileMesh = folderMesh + "RVE_POD_{0}.{1}"

opModel = 'Lin'
createMesh = False

contrast = 10.0
E2 = 1.0
E1 = contrast*E2 # inclusions
nu1 = 0.3
nu2 = 0.3

mu1 = elut.eng2mu(nu1,E1)
lamb1 = elut.eng2lambPlane(nu1,E1)
mu2 = elut.eng2mu(nu2,E2)
lamb2 = elut.eng2lambPlane(nu2,E2)

eps = np.zeros((2,2))
eps[0,0] = 1.0

param = np.array([[lamb1, mu1], [lamb2,mu2],[lamb1, mu1], [lamb2,mu2]])

maxOffset = 2

Lx = Ly = 1.0
ifPeriodic = False 
NxL = NyL = 2
Nx = (NxL+2*maxOffset)
Ny = (NyL+2*maxOffset)
NL = NxL*NyL
x0L = y0L = maxOffset*Lx/Nx
LxL = LyL = NxL*(x0L/maxOffset)
r0 = 0.2*LxL/NxL
r1 = 0.4*LxL/NxL
lcar = 0.1*LxL/NxL
NpLx = int(Lx/lcar) + 1
NpLxL = int(LxL/lcar) + 1
Vfrac = 0.282743

H = Lx/Nx

g = gmts.displacementGeneratorBoundary(x0L,y0L,LxL, LyL, NpLxL)

outerRadius = np.array([5.086605330529774677e-02, 5.672768852607592421e-02, 4.248077234944107328e-02, 4.947613702127881541e-02, 6.448002090833203359e-02,
6.462974709930800754e-02, 3.788142409556705809e-02, 4.008738015125086485e-02, 3.598442138610764146e-02, 4.713226785671519037e-02,
3.544989779856099615e-02, 4.765866308274066543e-02, 5.445442563985196383e-02, 4.211670407835125390e-02, 3.530535784797406151e-02,
5.115089010475230846e-02, 4.155890678105288172e-02, 4.629981581553022779e-02, 4.712063479976451308e-02, 3.871178940021258175e-02,
5.064971935987587492e-02, 5.963750994702962660e-02, 4.293841060126617204e-02, 4.049835030666170538e-02, 4.543728129541888677e-02,
6.645090773495036796e-02, 6.830071492806109867e-02, 5.533870691399940533e-02, 6.492343649246473669e-02, 6.240476224900730340e-02, 
4.512413568519717255e-02, 3.701531397986362187e-02])

seed0 = 40
ns = 20 

os.system('rm ' + folder + 'EpsList_{0}.hd5'.format(opModel))
os.system('rm ' + folder + 'sigmaL_{0}.hd5'.format(opModel))
os.system('rm ' + folder + 'sigmaT_{0}.hd5'.format(opModel))

EpsList, fEps = myhd.zeros_openFile(folder + 'EpsList_{0}.hd5'.format(opModel), (ns,maxOffset + 1,4)  , 'EpsList', mode = 'w')
SigmaL_list, fsigL = myhd.zeros_openFile(folder + 'sigmaL_{0}.hd5'.format(opModel), (ns,maxOffset + 1,4) , 'SigmaL',  mode = 'w')
SigmaT_list, fsigT = myhd.zeros_openFile(folder + 'sigmaT_{0}.hd5'.format(opModel), (ns,maxOffset + 1,4)  , 'SigmaT' , mode = 'w')

for seed in range(ns):
    
    np.random.seed(seed + seed0)
    ellipseData, permTotal, permBox = geni.circularRegular2Regions(r0, r1, NxL, NyL, Lx, Ly, maxOffset, ordered = False)
    ellipseData[permBox[4:],2] = outerRadius
    ellipseData = ellipseData[permTotal]
    
    # reescaling (but taking the outer as it is)
    for i in range(1): # should be maxOffset
        ni =  (NxL + 2*(i-1))*(NxL + 2*(i-1))
        nout = (NxL + 2*i)*(NxL + 2*i) 
        alphaFrac = np.sqrt(((nout-ni)*H**2)*Vfrac/(np.pi*np.sum(ellipseData[ni:nout,2]**2)))
        ellipseData[ni:nout,2] = alphaFrac*ellipseData[ni:nout,2]
        
    np.savetxt(folder + 'ellipseData_{0}.txt'.format(seed), ellipseData)
    
    meshGMSHred = gmsh.ellipseMesh2(ellipseData[:NL], x0L, y0L, LxL, LyL , lcar, ifPeriodic)
    meshGMSHred.setTransfiniteBoundary(NpLxL)
    meshRed = feut.getMesh(meshGMSHred, 'reduced_{0}'.format(seed), radFileMesh, createMesh)    
    
    for offset in range(maxOffset + 1):
                     
        Nt = (NxL + 2*offset)**2
        Lxt = Lyt =  H*np.sqrt(Nt)
        NpLxt = int(Lxt/lcar) + 1
         
        x0 = x0L - offset*H; y0 = y0L - offset*H
        
        if(offset == 0):
            meshGMSH = meshGMSHred
            mesh = meshRed
        else:
            meshGMSH = gmsh.ellipseMesh2Domains(x0L, y0L, LxL, LyL, NL, ellipseData[:Nt], Lxt, Lyt, lcar, x0 = x0, y0 = y0)
            meshGMSH.setTransfiniteBoundary(NpLxt)
            meshGMSH.setTransfiniteInternalBoundary(NpLxL)
            mesh = feut.getMesh(meshGMSH, 'offset{0}_{1}'.format(offset,seed), radFileMesh, createMesh)
    
    
        BM = fmts.getBMrestriction(g, mesh)        
    
        sigma, sigmaEps = fmts.getSigma_SigmaEps(param,mesh,eps)
        
        # Solving with Multiphenics
        U = mpms.solveMultiscale(param, mesh, eps, op = opModel, others = [x0, x0 + Lxt, y0, y0 + Lyt])
        T, a, B = feut.getAffineTransformationLocal(U[0],mesh,[0,1])
        Eps = - B.flatten()
    
        Utranslated = U[0] + T
    
        sigma_T = fmts.homogenisation(U[0], mesh, sigma, [0,1,2,3], sigmaEps).flatten()
        sigma_L = fmts.homogenisation(U[0], mesh, sigma, [0,1], sigmaEps).flatten()
                
        epsRed = -B + eps
        
        sigmaRed, sigmaEpsRed = fmts.getSigma_SigmaEps(param[0:2,:],meshRed,epsRed)
        Ured = mpms.solveMultiscale(param[0:2,:], meshRed, epsRed, op = 'BCdirich_lag', others = [Utranslated])
        
        print('sigma_L:', sigma_L)
        
        os.system("rm " + radFile.format('solRed_{0}_offset{1}_{2}'.format(opModel,offset,seed),'h5'))
        
        with HDF5File(MPI.comm_world, radFile.format('solRed_{0}_offset{1}_{2}'.format(opModel,offset,seed),'h5'), 'w') as f:
            f.write(Ured[0], 'basic')
                
        iofe.postProcessing_complete(U[0], folder + 'sol_mesh_{0}_offset{1}_{2}.xdmf'.format(opModel,offset,seed), ['u','lame','vonMises'], param)
        
        EpsList[seed, offset, :] = Eps
        SigmaL_list[seed, offset, :] = sigma_L
        SigmaT_list[seed, offset, :] = sigma_T
            
   
fEps.close() 
fsigL.close()
fsigT.close()                