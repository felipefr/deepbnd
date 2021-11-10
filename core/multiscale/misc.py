import dolfin as df
import numpy as np
from core.fenics_tools.misc import (Integral, VecX_expression,
affineTransformationExpression)


# Maps given tangents into the spatially varying elasticity tensor 
# piecewise continous
class Chom_multiscale(df.UserExpression):
    def __init__(self, tangent, mapping,  **kwargs):
        self.tangent = tangent
        self.map = mapping
        
        super().__init__(**kwargs)
        
    def eval_cell(self, values, x, cell):
        values[:] = self.tangent[self.map[cell.index],:,:].flatten()
        
    def value_shape(self):
        return (3,3,)

# Transform fluctuations to match zero-averages constraints (Minimally Const.)
def getAffineTransformationLocal(U,mesh, domains_id = [], justTranslation = False):
    dxList = [mesh.dx(i) for i in domains_id]
    omegaL = sum([df.assemble(df.Constant(1.0)*dxi) for dxi in dxList])
    
    yG = Integral(VecX_expression() , dxList, (2,))/omegaL

    uFlucL = Integral(U, dxList, (2,))/omegaL
    epsFlucL = Integral(df.grad(U), dxList, (2,2))/omegaL
    
    a = -uFlucL + epsFlucL@yG 
    B = np.zeros((2,2)) if justTranslation else -epsFlucL
    
    return affineTransformationExpression(a,B, mesh) , a, B

# Transform fluctuations to match zero-averages constraints (Minimally Const.)
# using u otimes n integral (no need of domain mesh)
def getAffineTransformationLocal_bndIntegral(U,mesh, domains_id = [], justTranslation = False):
    omegaL = df.assemble(df.Constant(1.0)*mesh.dx)
    
    yG = Integral(VecX_expression() , mesh.dx, (2,))/omegaL

    n = df.FacetNormal(mesh)
    uFlucL = Integral(U, mesh.dx, (2,))/omegaL
    epsFlucL = Integral(df.outer(U,n), mesh.ds, (2,2))/omegaL
    
    a = -uFlucL + epsFlucL@yG 
    B = np.zeros((2,2)) if justTranslation else -epsFlucL
    
    return affineTransformationExpression(a,B, mesh) , a, B


