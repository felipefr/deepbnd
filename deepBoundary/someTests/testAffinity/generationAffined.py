import sys, os
from numpy import isclose
from dolfin import *
# import matplotlib.pyplot as plt
from multiphenics import *
sys.path.insert(0, '../../../utils/')

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

folder = "./"
radFile = folder + "RVE_POD_{0}.{1}"

np.random.seed(10)

opModel = 'MR'

offset = 2
Lx = Ly = 1.0
ifPeriodic = False 
NxL = NyL = 2
NL = NxL*NyL
x0L = y0L = offset*Lx/(NxL+2*offset)
LxL = LyL = offset*Lx/(NxL+2*offset)
r0 = 0.2*LxL/NxL
r1 = 0.4*LxL/NxL
times = 1
lcar = 0.1*LxL/NxL
NpLx = int(Lx/lcar) + 1
NpLxL = int(LxL/lcar) + 1
Vfrac = 0.282743

print("nodes per side",  NpLx)
print("nodes per internal side",  NpLxL )

contrast = 10.0
E2 = 1.0
E1 = contrast*E2 # inclusions
nu1 = 0.3
nu2 = 0.3

mu1 = elut.eng2mu(nu1,E1)
lamb1 = elut.eng2lambPlane(nu1,E1)
mu2 = elut.eng2mu(nu2,E2)
lamb2 = elut.eng2lambPlane(nu2,E2)

param = np.array([[lamb1, mu1], [lamb2,mu2],[lamb1, mu1], [lamb2,mu2]])

g = gmts.displacementGeneratorBoundary(x0L,y0L,LxL, LyL, NpLxL)

seed = 1
np.random.seed(seed)

ellipseData1 = geni.circularRegular2Regions(r0, r1, NxL, NyL, Lx, Ly, offset, ordered = True)[NL:,:]
betaFrac = np.sqrt((1.0 - LxL*LyL)*Vfrac/(np.pi*np.sum(ellipseData1[:,2]*ellipseData1[:,2])))
ellipseData1[:,2] = betaFrac*ellipseData1[:,2]

ellipseData2 = geni.circularRegular2Regions(r0, r1, NxL, NyL, Lx, Ly, offset, ordered = True)[:NL,:]
alphaFrac = np.sqrt((LxL*LyL)*Vfrac/(np.pi*np.sum(ellipseData2[:,2]*ellipseData2[:,2])))
ellipseData2[:,2] = alphaFrac*ellipseData2[:,2]
        

ellipseData = np.concatenate((ellipseData2,ellipseData1),axis = 0)
        
meshGMSH = gmsh.ellipseMesh2Domains(x0L, y0L, LxL, LyL, NL, ellipseData, Lx, Ly , lcar, ifPeriodic)
meshGMSH.setTransfiniteBoundary(NpLx)
meshGMSH.setTransfiniteInternalBoundary(NpLxL)
        
meshGMSHred = gmsh.ellipseMesh2(ellipseData2, x0L, y0L, LxL, LyL , lcar, ifPeriodic)
meshGMSHred.setTransfiniteBoundary(NpLxL)
        
mesh = feut.getMesh(meshGMSH, '' , radFile)
meshRed = feut.getMesh(meshGMSHred, 'reduced', radFile)

W = VectorFunctionSpace(mesh,"CG", 1)
V = VectorFunctionSpace(meshRed,"CG", 1)
nodes = g.identifyNodes(meshRed)

dofsBoundary = np.zeros(2*len(nodes),'int')

v2d = vertex_to_dof_map(V) # used to take a vector in its dofs and transform to dofs according to node numbering in mesh
v2dW = vertex_to_dof_map(W) # used to take a vector in its dofs and transform to dofs according to node numbering in mesh

dofsBoundary[0::2] = v2d[2*nodes] 
dofsBoundary[1::2] = v2d[2*nodes + 1]

BM = fmts.getBMrestriction(g, mesh)        

Neps = 3
epsList = [np.zeros((2,2)),np.zeros((2,2)),np.zeros((2,2))]
epsList[0][0,0] = 1.0
epsList[1][1,1] = 1.0
epsList[2][1,0] = 0.5
epsList[2][0,1] = 0.5

Eps = np.zeros((Neps,4))
sigmaList = np.zeros((Neps,4))

sigma = []
sigmaEps = []

epsRedList = []

sigmaRed = []
sigmaRedEps = []

S = []
SW = []

for i in range(Neps):
    sigmaTemp = fmts.getSigma_SigmaEps(param,mesh,epsList[i])
    sigma.append(sigmaTemp[0])
    sigmaEps.append(sigmaTemp[1])
    

    # Solving with Multiphenics
    U = mpms.solveMultiscale(param, mesh, epsList[i], op = opModel)
    T, a, B = feut.getAffineTransformationLocal(U[0],mesh,[0,1])
    Eps[i,:] = - B.flatten()
    
    sigma_MR = fmts.homogenisation(U[0], mesh, sigma[i], [0,1], sigmaEps[i]).flatten()
            
    epsRedList.append(-B + epsList[i])
    
    sigmaRedTemp = fmts.getSigma_SigmaEps(param[0:2,:],meshRed,epsRedList[i])
    sigmaRed.append(sigmaRedTemp[0])
    sigmaRedEps.append(sigmaRedTemp[1])
    
    uBoundary = Function(V)
    uBoundary.vector()[dofsBoundary] = g(mesh,U[0] + T).flatten()
    
    Ured = mpms.solveMultiscale(param[0:2,:], meshRed, epsRedList[i], op = 'BCdirich_lag', others = [uBoundary])
    sigma_red_MR = fmts.homogenisation(Ured[0], meshRed, sigmaRed[i], [0,1], sigmaRedEps[i]).flatten()
    
    print('sigma_MR:', sigma_MR)
    print('sigma_red_MR:', sigma_red_MR)
    print('sigma_MR - sigma_red_MR:', sigma_MR - sigma_red_MR)
    
    sigmaList[i,:] = sigma_red_MR
    
    v2dW = vertex_to_dof_map(U[0].function_space())

    
    S.append(Ured[0].vector()[v2d])
    SW.append(U[0].vector()[v2dW])
   
S = np.array(S).T
SW = np.array(SW).T

# Test for the entire RVE
epsV = np.array([[0.1,-0.8],[0.5,2.0]])
Uv = mpms.solveMultiscale(param, mesh, epsV, op = opModel)

sigmaV_, sigmaV_Eps= fmts.getSigma_SigmaEps(param,mesh,epsV)
sigmaV = fmts.homogenisation(Uv[0], mesh, sigmaV_, [0,1], sigmaV_Eps).flatten()
   

v2dW = vertex_to_dof_map(Uv[0].function_space())
Uv_dofs = Uv[0].vector()[v2dW]

Uv_dofs_rec = SW@elut.tensor2voigt_sym(epsV)
Uv_rec = Function(W)
v2dW = vertex_to_dof_map(W)
Uv_rec.vector()[v2dW] = Uv_dofs_rec[:]


sigmaV_rec = fmts.homogenisation(Uv_rec, mesh, sigmaV_, [0,1], sigmaV_Eps).flatten()


print(np.linalg.norm(Uv_dofs - Uv_dofs_rec))

print(np.linalg.norm(sigmaV_rec -sigmaV))

# Test for the small portion of the RVE

T, a, B = feut.getAffineTransformationLocal(Uv_rec,mesh,[0,1])
epsL = - B + epsV
epsL = 0.5*(epsL + epsL.T)

uBoundary_v = Function(V)
uBoundary_v.vector()[dofsBoundary] = g(mesh,Uv_rec + T).flatten()
    
Uv_red = mpms.solveMultiscale(param[0:2,:], meshRed, epsL, op = 'BCdirich_lag', others = [uBoundary_v])

sigmaV_L_, sigmaV_Eps_L= fmts.getSigma_SigmaEps(param[0:2,:],meshRed,epsL)
sigmaV_L = fmts.homogenisation(Uv_red[0], meshRed, sigmaV_L_, [0,1], sigmaV_Eps_L).flatten()
   

v2dW = vertex_to_dof_map(Uv_red[0].function_space())
Uv_red_dofs = Uv_red[0].vector()[v2dW]

Uv_red_dofs_rec = S@elut.tensor2voigt_sym(epsV)
Uv_red_rec = Function(V)
v2d = vertex_to_dof_map(V)
Uv_red_rec.vector()[v2d] = Uv_red_dofs_rec[:]


sigmaV_L_rec = fmts.homogenisation(Uv_red_rec, meshRed, sigmaV_L_, [0,1], sigmaV_Eps_L).flatten()


print(np.linalg.norm(Uv_red_dofs - Uv_red_dofs_rec))

print(np.linalg.norm(sigmaV_L_rec -sigmaV_L))


print(np.linalg.norm(sigmaV_L -sigmaV))
print(np.linalg.norm(sigmaV_L_rec -sigmaV))
