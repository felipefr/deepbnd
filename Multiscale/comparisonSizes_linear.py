#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 14:15:46 2020

@author: felipefr
"""


import numpy as np
import sys
sys.path.insert(0, '../utils/')
from timeit import default_timer as timer
import elasticity_utils as elut
import ioFenicsWrappers as iofe
import Snapshots as snap
import sys, os
sys.path.insert(0, '../utils/')
import Generator as gene
import genericParam as gpar
import fenicsWrapperElasticity as fela
import wrapperPygmsh as gmsh
# import pyvista as pv
import generationInclusions as geni
import ioFenicsWrappers as iofe
import fenicsMultiscale as fmts
import dolfin as df
import generatorMultiscale as gmts
import fenicsUtils as feut

param = np.array([ [10.0,15.0], [20.0,30.0], [10.0,15.0], [20.0,30.0] ])
        
eps = np.zeros((2,2))
eps[0,0] = 0.1

Lx = Ly = 1.0
ifPeriodic = False 
NxL = NyL = 1

np.random.seed(10)

ns = 50
maxOffset = 5
SigmaListL = np.zeros((ns,maxOffset-1, 4))
SigmaList = np.zeros((ns,maxOffset-1, 4))
uBoundList = np.zeros((ns,maxOffset-1,160))


for offset in range(1,maxOffset):
    x0L = y0L = offset*Lx/(1+2*offset)
    LxL = LyL = Lx/(1+2*offset)
    r0 = 0.3*LxL/NxL
    r1 = 0.4*LxL/NxL
    lcar = LxL/20
 
    for k in range(ns): 
        ellipseData = geni.circularRegular2Regions(r0, r1, NxL, NyL, Lx, Ly, offset, ordered = True)
        if(offset<4):
            continue
        ellipseData[0,2] = 0.5*(r0 + r1)
        meshGMSH =  gmsh.ellipseMesh2DomainsPhysicalMeaning(x0L, y0L, LxL, LyL, NxL*NyL, ellipseData , Lx, Ly, lcar, ifPeriodic) 

        meshGeoFile = 'test_' + str(offset) + '_' + str(k) + '.geo'
        meshGMSH.write(meshGeoFile,'geo')
        os.system('gmsh -2 -algo del2d -format msh2 ' + meshGeoFile)
    
        os.system('dolfin-convert ' + meshGeoFile[:-3] + 'msh ' + meshGeoFile[:-3] + 'xml')
        
        meshFenics = fela.EnrichedMesh(meshGeoFile[:-3] + 'xml')
        meshFenics.createFiniteSpace('V', 'u', 'CG', 1)
                
        sigmaEps = fmts.getSigmaEps(param, meshFenics, eps)
        vol = df.assemble(df.Constant(1.0)*meshFenics.dx)
        
        # Traditional way: mixed MR multiscale
        u_trad = fmts.solveMultiscaleLinear(param, meshFenics, eps)
        
        sigmaL_trad = fmts.homogenisation_noMesh(u_trad, fmts.getSigma(param,meshFenics), [0,1], sigmaEps)
        sigma_trad = fmts.homogenisation_noMesh(u_trad, fmts.getSigma(param,meshFenics), [0,1,2,3], sigmaEps)
        
        # sigma_LM = feut.Integral(P, meshFenics.dx, shape = (2,2))/vol
        
        SigmaListL[k,offset - 1, : ] = sigmaL_trad.reshape((4,))
        SigmaList[k,offset - 1, : ] = sigma_trad.reshape((4,))
        
        g = gmts.displacementGeneratorBoundary(x0L,y0L,LxL,LyL,21)
        uBound = g(meshFenics,u_trad)  
        uBoundList[k,offset-1, :] = uBound[:,0]
        
        # iofe.postProcessing_complete(u_trad, "solution_trad.xdmf", ['u', 'lame', 'vonMises'], param , rename = False)

for offset in range(4,maxOffset):
    np.savetxt('SigmaListL_linear_' + str(offset) + '.txt', SigmaListL[:,offset-1,:])
    np.savetxt('SigmaList_linear_' + str(offset) + '.txt', SigmaList[:,offset-1,:])
    np.savetxt('uBoundList_linear_' + str(offset) + '.txt', uBoundList[:,offset-1,:])
