#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 07:18:50 2022

@author: felipe
"""

import sys, os
import numpy as np
import dolfin as df 
import matplotlib.pyplot as plt
from ufl import nabla_div
from timeit import default_timer as timer
import multiphenics as mp
from mpi4py import MPI

from deepBND.__init__ import *
import micmacsfenics.core.micro_constitutive_model as mscm
from deepBND.core.fenics_tools.enriched_mesh import EnrichedMesh 
from deepBND.core.elasticity.fenics_utils import getLameInclusions
from deepBND.core.fenics_tools.misc import symgrad, Integral
from deepBND.core.elasticity.misc import stress2voigt, strain2voigt, voigt2strain, voigt2stress
import deepBND.core.fenics_tools.misc as feut

comm = MPI.COMM_WORLD
comm_self = MPI.COMM_SELF
rank = comm.Get_rank()
num_ranks = comm.Get_size()

class MicroConstitutiveModelGen(mscm.MicroConstitutiveModel):
    
    def __init__(self,nameMesh, param, model):
        self.nameMesh = nameMesh
        self.param = param
        self.model = model
        # it should be modified before computing tangent (if needed) 
        self.others = {'polyorder' : 2} 
        
        self.multiscaleModel = mscm.listMultiscaleModels[model]              
        self.ndim = 2
        self.nvoigt = int(self.ndim*(self.ndim + 1)/2)  
 
        self.getHomogenisation = self.computeHomogenisation # in the first run should compute     
        
        self.Hom = {"sigma": np.zeros((self.nvoigt, self.nvoigt)),
               "sigmaL": np.zeros((self.nvoigt, self.nvoigt)),
               "eps": np.zeros((self.nvoigt, self.nvoigt)),
               "epsL": np.zeros((self.nvoigt, self.nvoigt)),
               "tangent": np.zeros((self.nvoigt, self.nvoigt)),
               "tangentL": np.zeros((self.nvoigt, self.nvoigt))} 
        
    def readMesh(self):
        self.mesh = EnrichedMesh(self.nameMesh,comm_self)
        self.lame = getLameInclusions(*self.param, self.mesh)
        self.coord_min = np.min(self.mesh.coordinates(), axis = 0)
        self.coord_max = np.max(self.mesh.coordinates(), axis = 0)
        self.others['x0'] = self.coord_min[0]
        self.others['x1'] = self.coord_max[0] 
        self.others['y0'] = self.coord_min[1]
        self.others['y1'] = self.coord_max[1]
    
    def computeHomogenisation(self, domainL = [0, 1]):    # homogenisation of a lot of things, sometimes checks  
        
        self.readMesh()
        sigmaLaw = lambda u: self.lame[0]*nabla_div(u)*df.Identity(2) + 2*self.lame[1]*symgrad(u)
        
        dy = self.mesh.dx # specially for the case of enriched mesh, otherwise it does not work
        volL = sum([ df.assemble(df.Constant(1.0)*dy(i)) for i in domainL])
        vol = df.assemble(df.Constant(1.0)*dy)
        
        y = df.SpatialCoordinate(self.mesh)
        yG_L = sum([Integral(y, dy(i), (2,)) for i in domainL])/volL
        yG = Integral(y, dy, (2,))/vol
        
        assert( np.linalg.norm(yG_L) < 1.0e-9)
        assert( np.linalg.norm(yG) < 1.0e-9)
        
        yG_L = df.Constant(yG_L)
        yG = df.Constant(yG)

        Eps = df.Constant(((0.,0.),(0.,0.))) # just placeholder
        
        form = self.multiscaleModel(self.mesh, sigmaLaw, Eps, self.others)
        a,f,bcs,W = form()

        start = timer()        
        A = mp.block_assemble(a)
        if(len(bcs) > 0): 
            bcs.apply(A)
        
        # solver = df.PETScLUSolver('superlu')
        solver = df.PETScLUSolver()
        sol = mp.BlockFunction(W)
        
        if(self.model == 'dnn'):
            Vref = self.others['uD'].function_space()
            Mref = Vref.mesh()
            normal = df.FacetNormal(Mref)
        
        B = np.zeros((2,2))
        
        
        for i in range(self.nvoigt):
            
            
            start = timer()              
            if(self.model == 'dnn'):
                self.others['uD'].vector().set_local(self.others['uD{0}_'.format(i)])
            
                B = -Integral(df.outer(self.others['uD'],normal), Mref.ds, (2,2))/volL
                B = 0.5*(B + B.T)
                
                T = feut.affineTransformationExpression(np.zeros(2),B, Mref) # ignore a, since the basis is already translated
                self.others['uD'].vector().set_local(self.others['uD'].vector().get_local()[:] + 
                                                     df.interpolate(T,Vref).vector().get_local()[:])
                
                         
            Eps.assign(df.Constant(mscm.macro_strain(i) - B))   
        
            F = mp.block_assemble(f)
            if(len(bcs) > 0):
                bcs.apply(F)
        
            solver.solve(A,sol.block_vector(), F)    
            sol.block_vector().block_function().apply("to subfunctions")
            
            u_mu = df.dot(Eps, y - yG) + sol[0]
            eps_mu = symgrad(u_mu) # the value of yG doest not matter for the stress/strain
            
            sig_mu = sigmaLaw(u_mu) # symmetry guaranteed
            
            sigmaL_hom =  sum([Integral(sig_mu, dy(i), (2,2)) for i in domainL])/volL
            sigma_hom =  Integral(sig_mu, dy, (2,2))/vol
            
            epsL_hom =  sum([Integral(eps_mu, dy(i), (2,2)) for i in domainL])/volL
            eps_hom =  Integral(eps_mu, dy, (2,2))/vol
            
            self.Hom['sigmaL'][:,i] = stress2voigt(sigmaL_hom)
            self.Hom['sigma'][:,i] = stress2voigt(sigma_hom)
            self.Hom['epsL'][:,i] = strain2voigt(epsL_hom)
            self.Hom['eps'][:,i] = strain2voigt(eps_hom)
            
            
            end = timer()
            print('time in solving system', end - start) # Time in seconds
        
        
        self.Hom['tangent'] = self.Hom['sigma'] @ np.linalg.inv(self.Hom['eps'])
        self.Hom['tangentL'] = self.Hom['sigmaL'] @ np.linalg.inv(self.Hom['epsL'])
        
        self.getHomogenisation = self.getHomogenisation_ # from the second run onwards, just returns  
                
        return self.Hom
        
    
    def getHomogenisation_(self):
        return self.Hom

