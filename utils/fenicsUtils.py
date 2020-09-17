import dolfin as df
import numpy as np
import os
import fenicsWrapperElasticity as fela

def Integral(u,dx,shape):
    
    n = len(shape)
    I = np.zeros(shape)
    
    if(type(dx) != type([])):
        dx = [dx]
 
    if(n == 1):
        for i in range(shape[0]):
            for dxj in dx:
                I[i] += df.assemble(u[i]*dxj)
            
    elif(n == 2):
        for i in range(shape[0]):
            for j in range(shape[1]):
                for dxk in dx:
                    I[i,j] += df.assemble(u[i,j]*dxk)
    
    else:
        print('not implement for higher order integral')
        
    
    return I
    
def getMesh(meshGMSH, label, radFile, create = False):

    meshXmlFile = radFile.format(label,'xml')
    
    if(create):
        meshGeoFile = radFile.format(label,'geo')
        meshMshFile = radFile.format(label,'msh')
        meshGMSH.write(meshGeoFile,'geo')
        os.system('gmsh -2 -algo del2d -format msh2 ' + meshGeoFile)
        
        os.system('dolfin-convert {0} {1}'.format(meshMshFile, meshXmlFile))
        
    return fela.EnrichedMesh(meshXmlFile)


def affineTransformationExpession(a,B, mesh):
    return df.Expression(('a0 + B00*x[0] + B01*x[1]','a1 + B10*x[0] + B11*x[1]'), a0 = a[0], a1 = a[1],
               B00=B[0,0], B01 = B[0,1], B10 = B[1,0], B11= B[1,1] ,degree = 1, domain = mesh)

    
def getAffineTransformationLocal(U,mesh, domains_id = []):
    dxList = [mesh.dx(i) for i in domains_id]
    omegaL = sum([df.assemble(df.Constant(1.0)*dxi) for dxi in dxList])
    
    yG = Integral(df.Expression(('x[0]','x[1]'), degree = 1) , dxList, (2,))/omegaL
    uFlucL = Integral(U, dxList, (2,))/omegaL
    epsFlucL = Integral(df.grad(U), dxList, (2,2))/omegaL
    
    a = -uFlucL + epsFlucL@yG 
    B = -epsFlucL

    return affineTransformationExpession(a,B, mesh) , a, B
