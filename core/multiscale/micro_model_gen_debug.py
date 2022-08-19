#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 13:24:11 2022

@author: felipefr
"""

import numpy as np
import dolfin as df 
from ufl import nabla_div
from timeit import default_timer as timer
import multiphenics as mp
from mpi4py import MPI

from deepBND.__init__ import *
import micmacsfenics.core.micro_constitutive_model as mscm
from fetricks.fenics.mesh.mesh import Mesh 
from fetricks.fenics.material.multimaterial import getLameExpression
import fetricks as ft 
from fetricks.fenics.misc import affineTransformationExpression 

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
        self.mesh = Mesh(self.nameMesh,comm_self)
        self.lame = getLameExpression(*self.param, self.mesh)
        self.coord_min = np.min(self.mesh.coordinates(), axis = 0)
        self.coord_max = np.max(self.mesh.coordinates(), axis = 0)
        self.others['x0'] = self.coord_min[0]
        self.others['x1'] = self.coord_max[0] 
        self.others['y0'] = self.coord_min[1]
        self.others['y1'] = self.coord_max[1]
    
    def computeHomogenisation(self, domainL = [0, 1]):    # homogenisation of a lot of things, sometimes checks  
        
        self.readMesh()
        sigmaLaw = lambda e: self.lame[0]*ft.tr_mandel(e)*ft.Id_mandel_df + 2*self.lame[1]*e
        sigmaLaw_u = lambda u: ft.mandel2tensor( sigmaLaw(ft.symgrad_mandel(u)) )
        
        dy = self.mesh.dx # specially for the case of enriched mesh, otherwise it does not work
        volL = sum([ df.assemble(df.Constant(1.0)*dy(i)) for i in domainL])
        vol = df.assemble(df.Constant(1.0)*dy)
        
        y = df.SpatialCoordinate(self.mesh)
        yG_L = sum([ft.Integral(y, dy(i), (2,)) for i in domainL])/volL
        yG = ft.Integral(y, dy, (2,))/vol
        
        assert( np.linalg.norm(yG_L) < 1.0e-9)
        assert( np.linalg.norm(yG) < 1.0e-9)
        
        yG_L = df.Constant(yG_L)
        yG = df.Constant(yG)

        Eps = df.Constant(((0.,0.),(0.,0.))) # just placeholder
        
        form = self.multiscaleModel(self.mesh, sigmaLaw_u, Eps, self.others)
        a,f,bcs,W = form()

        start = timer()        
        A = mp.block_assemble(a)
        if(len(bcs) > 0): 
            bcs.apply(A)
        
        solver = df.PETScLUSolver('mumps')
        #solver = df.PETScLUSolver()
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
            
                # B = - ft.Integral(df.outer(self.others['uD'],normal), Mref.ds, (2,2))/volL
                # print("B = ", B)
                # B.fill(0.0)
                # B = 0.5*(B + B.T)
                
                # T = affineTransformationExpression(np.zeros(2),B, Mref) # ignore a, since the basis is already translated
                # self.others['uD'].vector().set_local(self.others['uD'].vector().get_local()[:] + 
                                                     # df.interpolate(T,Vref).vector().get_local()[:])
                
                         
            Eps.assign(df.Constant(mscm.macro_strain(i)))   
        
            F = mp.block_assemble(f)
            if(len(bcs) > 0):
                bcs.apply(F)
        
            solver.solve(A,sol.block_vector(), F)    
            sol.block_vector().block_function().apply("to subfunctions")
            
            u_mu = df.dot(Eps, y - yG) + sol[0]
            eps_mu = ft.symgrad_mandel(u_mu) # the value of yG doest not matter for the stress/strain
            
            sig_mu = sigmaLaw(eps_mu) # symmetry guaranteed
            
            sigmaL_hom =  sum([ft.Integral(sig_mu, dy(i), (self.nvoigt,)) for i in domainL])/volL
            sigma_hom =  ft.Integral(sig_mu, dy, (self.nvoigt,))/vol
            
            epsL_hom =  sum([ft.Integral(eps_mu, dy(i), (self.nvoigt,)) for i in domainL])/volL
            eps_hom =  ft.Integral(eps_mu, dy, (self.nvoigt,))/vol
            
            self.Hom['sigmaL'][:,i] = ft.mandel2voigtStress(sigmaL_hom)
            self.Hom['sigma'][:,i] = ft.mandel2voigtStress(sigma_hom)
            self.Hom['epsL'][:,i] = ft.mandel2voigtStrain(epsL_hom)
            self.Hom['eps'][:,i] = ft.mandel2voigtStrain(eps_hom)
            
            
            end = timer()
            print('time in solving system', end - start) # Time in seconds
        
        
        self.Hom['tangent'] = self.Hom['sigma'] @ np.linalg.inv(self.Hom['eps'])
        self.Hom['tangentL'] = self.Hom['sigmaL'] @ np.linalg.inv(self.Hom['epsL'])
        
        self.getHomogenisation = self.getHomogenisation_ # from the second run onwards, just returns  
                
        return self.Hom
        
    
    def getHomogenisation_(self):
        return self.Hom

