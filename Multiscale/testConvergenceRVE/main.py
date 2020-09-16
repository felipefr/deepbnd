import sys, os
import numpy as np
sys.path.insert(0, '../../utils/')
from timeit import default_timer as timer
import ioFenicsWrappers as iofe
import Snapshots as snap
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
import elasticity_utils as elut

op = 'periodic'

folder = "./simuls_10/"
radFile = folder + "test_{0}_{1}.{2}"

contrast = 10.0
E2 = 1.0
E1 = contrast*E2 # inclusions
nu1 = 0.3
nu2 = 0.3

mu1 = elut.eng2mu(nu1,E1)
lamb1 = elut.eng2lambPlane(nu1,E1)
mu2 = elut.eng2mu(nu2,E2)
lamb2 = elut.eng2lambPlane(nu2,E2)

param = np.array([[lamb1, mu1], [lamb2,mu2]])
        
eps = np.zeros((2,2))
eps[0,0] = 0.1

np.random.seed(10)

replications = 9

offset = 0
Lx = Ly = 1.0
ifPeriodic = False 
NxL = NyL = 2
x0L = y0L = offset*Lx/(1+2*offset)
LxL = LyL = Lx/(1+2*offset)
r0 = 0.2*LxL/NxL
r1 = 0.4*LxL/NxL

ellipseData = geni.circularRegular2Regions(r0, r1, NxL, NyL, Lx, Ly, offset, ordered = True)

for times in range(1, replications+1):
    lcar = 0.1*Lx/(NxL*times)
    
    meshGMSH = gmsh.ellipseMeshRepetition(times, ellipseData, Lx, Ly , lcar, ifPeriodic)
    meshGMSH.setTransfiniteBoundary(int(Lx/lcar) + 1)
    
    meshGeoFile = radFile.format(times,'','geo')
    meshXmlFile = radFile.format(times,'','xml')
    meshMshFile = radFile.format(times,'','msh')
    
    meshGMSH.write(meshGeoFile,'geo')
    os.system('gmsh -2 -algo del2d -format msh2 ' + meshGeoFile)
    
    os.system('dolfin-convert {0} {1}'.format(meshMshFile, meshXmlFile))
    
    meshFenics = fela.EnrichedMesh(meshXmlFile)
    meshFenics.createFiniteSpace('V', 'u', 'CG', 1)
            
    sigmaEps = fmts.getSigmaEps(param, meshFenics, eps)
    
    sol = fmts.solveMultiscale(param, meshFenics, eps, op, others = [[2]])
    if(op == 'linear'):
        u = sol
    else:
        u = sol.split()[0]
            
    sigma = fmts.homogenisation_noMesh(u, fmts.getSigma(param,meshFenics), [0,1], sigmaEps)
    solutionXdmfFile = radFile.format(op, times,'xdmf')
    iofe.postProcessing_complete(u, solutionXdmfFile, ['u', 'lame', 'vonMises'], param , rename = False)
    sigmaFile = radFile.format('sigma' , op + '_' + str(times), 'txt')

    with open(sigmaFile ,"a") as f:
        f.write("\n")
        np.savetxt(f,sigma.reshape((1,4)))
        
    Chom, Ehom, nu_hom  = fmts.homogenisedTangent(param, meshFenics, op, linBdr = [2])
    chomFile = radFile.format('Chom' , op + '_' + str(times), 'txt')
    np.savetxt(chomFile , np.array(Chom.reshape((9,)).tolist() + [Ehom, nu_hom]).reshape((1,11)))
    
    with open(chomFile ,"a") as f:
        f.write("\n")
        np.savetxt(f, np.array(Chom.reshape((9,)).tolist() + [Ehom, nu_hom]).reshape((1,11)))
        
