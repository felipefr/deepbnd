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

comm = MPI.COMM_WORLD
comm_self = MPI.COMM_SELF
rank = comm.Get_rank()
num_ranks = comm.Get_size()

class MicroModel(mscm.MicroConstitutiveModel):
    
    def __init__(self,nameMesh, param, model):
        self.nameMesh = nameMesh
        self.param = param
        self.model = model
        # it should be modified before computing tangent (if needed) 
        self.others = {'polyorder' : 2} 
        
        self.multiscaleModel = mscm.listMultiscaleModels[model]              
        self.ndim = 2
        self.nvoigt = int(self.ndim*(self.ndim + 1)/2)  
        self.sol = []

        self.readMesh()

    def readMesh(self):
        self.mesh = EnrichedMesh(self.nameMesh,comm_self)
        self.coord_min = np.min(self.mesh.coordinates(), axis = 0)
        self.coord_max = np.max(self.mesh.coordinates(), axis = 0)
        self.others['x0'] = self.coord_min[0]
        self.others['x1'] = self.coord_max[0] 
        self.others['y0'] = self.coord_min[1]
        self.others['y1'] = self.coord_max[1]
        self.y = df.SpatialCoordinate(self.mesh)
        self.dy = self.mesh.dx # specially for the case of enriched mesh, otherwise it does not work
        self.lame = getLameInclusions(*self.param, self.mesh)
        self.sigmaLaw = lambda u: self.lame[0]*nabla_div(u)*df.Identity(2) + 2*self.lame[1]*symgrad(u)

    
    def compute(self):  # maybe should be removed
                
        Eps = df.Constant(((0.,0.),(0.,0.))) # just placeholder
        
        form = self.multiscaleModel(self.mesh, self.sigmaLaw, Eps, self.others)
        a,f,bcs,W = form()

        start = timer()        
        A = mp.block_assemble(a)
        if(len(bcs) > 0): 
            bcs.apply(A)
        
        solver = df.PETScLUSolver('mumps')
        self.sol = [mp.BlockFunction(W),mp.BlockFunction(W),mp.BlockFunction(W)]
        
        for i in range(self.nvoigt):
        
            start = timer()              
                
            Eps.assign(df.Constant(mscm.macro_strain(i)))   
        
            F = mp.block_assemble(f)
            if(len(bcs) > 0):
                bcs.apply(F)
        
            solver.solve(A,self.sol[i].block_vector(), F)    
            self.sol[i].block_vector().block_function().apply("to subfunctions")
            
            end = timer()
            print('time in solving system', end - start) # Time in seconds
        
        
        
    def homogenise(self, regions = [0,1], i_voigt = 0 ): # maybe should be removed
        
        Eps = df.Constant(((0.,0.),(0.,0.))) # just placeholder
        Eps.assign(df.Constant(mscm.macro_strain(i_voigt)))   
    
        vol = sum([df.assemble(df.Constant(1.0)*self.dy(i)) for i in regions])
        
        sig_mu = self.sigmaLaw(df.dot(Eps,self.y) + self.sol[i_voigt][0])
        sigma_hom =  sum([Integral(sig_mu, self.dy(i), (2,2)) for i in regions])/vol
        

        return sigma_hom
            
