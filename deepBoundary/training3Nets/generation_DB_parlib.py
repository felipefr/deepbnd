import sys, os
from numpy import isclose
import fenics
from dolfin import *
import matplotlib.pyplot as plt
from multiphenics import *
sys.path.insert(0, '../../utils/')

import dill

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
import fenicsUtils as feut

from timeit import default_timer as timer

from mpi4py import MPI as pyMPI
import pickle
from commonParameters import *

from joblib import Parallel, delayed, parallel_backend
import funcs
import multiprocessing as mp
from functools import partial

comm = MPI.comm_world


def getBlockCorrelation_nosym(ib,ie,jb,je, dotProduct, radFile, radSolution):
    
    A = np.zeros((ie-ib, je-jb))
    Si = loadSimulations(ib, ie, radFile, radSolution)
    Sj = loadSimulations(jb, je, radFile, radSolution)
    
    for i,ii in enumerate(range(ib,ie)):
        V = Si[str(ii)].function_space()
        mesh = V.mesh()
        
        for j,jj in enumerate(range(jb,je)):                       
            ISj = interpolate(Sj[str(jj)],V)
            A[i,j] = dotProduct(S[str(ii)],ISj,mesh)

    return A

def getBlockCorrelation_sym(ib,ie, dotProduct, radFile, radSolution):
    
    A = np.zeros((ie-ib, ie-ib))
    S = loadSimulations(ib, ie, radFile, radSolution)
    
    for i in range(ie-ib):
        ii = ib + i
        V = S[str(ii)].function_space()
        mesh = V.mesh()
        
        for j in range(i,ie-ib):   
            jj = j + ib                    
            ISj = interpolate(Sj[str(jj)],V)
            
            A[i,j] = dotProduct(S[str(ii)],ISj,mesh)
            A[j,i] = A[i,j] 

    return A
    
    
def getCorrelation(A, Nblocks, dotProduct, radFile, radSolution):

    N = len(A)
    p = np.linspace(0,N,Nblocks+1).astype('int')

    NblocksTotal = int(Nblocks*(Nblocks + 1)/2)
    
    mapblock = []
    i = 0
    j = 0
    
    for k in range(NblocksTotal):
        mapblock.append((i,j))
        j+=1
        if(j>Nblocks-1):
            j = i + 1  
            i += 1
    
    # print(mapblock)
    
    # p = mp.Pool(4)
    
    # LOKY_PICKLER=dill
    
    with parallel_backend('loky', n_jobs=6):
        blocks = Parallel()(delayed(funcs.getBlockCorrelation_new)(k, mapblock, p, dotProduct, radFile, radSolution) for k in range(NblocksTotal))
    

    # foo = partial(funcs.getBlockCorrelation_new, mapblock = mapblock, p= p, dotProduct = dotProduct, 
    #               radFile = radFile, radSolution = radSolution)
    # blocks = p.map(foo, range(NblocksTotal))
     
    for k,b in enumerate(blocks):
        i, j = mapblock[k]
        ib = partitions[i]
        ie = partitions[i+1]
        jb = partitions[j]
        je = partitions[j+1]
        A[ib:ie,jb:je] = b[:,:]
        if(i != j):
            A[jb:je,ib:ie] = b[:,:].T