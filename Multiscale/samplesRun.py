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

folder = './simuls2/'
radical = folder

Lx = Ly = 1.0
lcar = 0.015
ifPeriodic = False 
NxL = NyL = 2
offset = 2
x0L = y0L = 1./3.0
LxL = LyL = 1./3.0
r0 = 0.2*LxL/NxL
r1 = 0.4*LxL/NxL

mu0 = elut.eng2mu(0.3,100.0)
mu1 = elut.eng2mu(0.3,10.0)
lamb0 = elut.eng2lambPlane(0.3,100.0)
lamb1 = elut.eng2lambPlane(0.3,10.0)

eps = np.zeros((2,2))
eps[0,0] = 0.1

param = np.array([ [lamb1,mu1], [lamb0,mu0], [lamb1,mu1], [lamb0,mu0] ])
femData = { 'meshFile' : '',
            'problem' : lambda x,y, z, w, s: fmts.solveMultiscale(x, y, z, w, s),
            'fespace' : {'spaceType' : 'V', 'name' : 'u', 'spaceFamily' : 'CG', 'degree' : 1} }
    
n = 20

sigma_hom = np.zeros((n,2,2))
sigma_homL = np.zeros((n,2,2))

np.random.seed(10)

for i in range(n):
    ellipseData = geni.circularRegular2Regions(r0, r1, NxL, NyL, Lx, Ly, offset, ordered = True)
    meshGMSH =  gmsh.ellipseMesh2DomainsPhysicalMeaning(x0L, y0L, LxL, LyL, NxL*NyL, ellipseData , Lx, Ly, lcar, ifPeriodic) 
    
    meshGMSH.write(folder + 'mesh_' + str(i) + '.geo','geo')
    os.system('gmsh -2 -algo del2d -format msh2 ' + folder + 'mesh_' + str(i) + '.geo')
    
    os.system('dolfin-convert ' + folder + 'mesh_' + str(i) + '.msh ' + folder + 'mesh_' + str(i) + '.xml')
    
    femData['meshFile'] = folder + 'mesh_' + str(i) + '.xml'

    meshFenics = fela.EnrichedMesh(femData['meshFile'])
    meshFenics.createFiniteSpace(**femData['fespace'])
    
    utot, u, sigma_hom[i,:,:], sigma_homL[i,:,:]  = femData['problem'](param, meshFenics, eps, [0,1,2,3], [0,1])
        
    iofe.postProcessing_complete(utot, folder + 'solution_tot_' + str(i) + '.xdmf', ['u','lame','vonMises'], param)
    iofe.postProcessing_complete(u, folder + 'solution_' + str(i) + '.xdmf', ['u','lame','vonMises'], param)


np.savetxt("sigmaL_2.txt", sigma_homL.reshape((n,4)))
np.savetxt("sigma_2.txt", sigma_hom.reshape((n,4)))