import numpy as np
import meshio
import pygmsh
import ioFenicsWrappers as iofe
import os
import dolfin as df

class EnrichedMesh(df.Mesh):
    def __init__(self, meshFile):
        super(EnrichedMesh,self).__init__()
        
        if(meshFile[-3:] == 'xml'):
            df.File(meshFile) >> self            
            self.subdomains = df.MeshFunction("size_t", self, meshFile[:-4] + "_physical_region.xml")
            self.boundaries = df.MeshFunction("size_t", self, meshFile[:-4] + "_facet_region.xml")
            
        elif(meshFile[-4:] == 'xdmf'):
            self.subdomains, self.boundaries = iofe.readXDMF_with_markers(meshFile, self)
                
        self.ds = df.Measure('ds', domain=self, subdomain_data=self.boundaries)
        self.dx = df.Measure('dx', domain=self, subdomain_data=self.subdomains)
            
        self.V = {}
        self.bcs = {}
        self.dsN = {}
        self.dxR = {}
        
    def createFiniteSpace(self,  spaceType = 'S', name = 'u', spaceFamily = 'CG', degree = 1):
        
        myFunctionSpace = df.TensorFunctionSpace if spaceType =='T' else (df.VectorFunctionSpace if spaceType == 'V' else df.FunctionSpace)
        
        self.V[name] = myFunctionSpace(self, spaceFamily, degree)
        
    def addDirichletBC(self, name = 'default', spaceName = 'u', g = df.Constant(0.0), markerLabel=0, sub = None):
        Vaux = self.V[spaceName] if type(sub)==type(None) else self.V[spaceName].sub(sub)
        self.bcs[name] = df.DirichletBC(Vaux, g , self.boundaries, markerLabel)
    
    def applyDirichletBCs(self,A,b = None):
        if(type(b) == type(None)):
            for bc in self.bcs.values():
                bc.apply(A)
        else:
            for bc in self.bcs.values():
                bc.apply(A,b)
                
    def nameNeumannBoundary(self, name, boundaryMarker):
        self.dsN[name] = reduce(lambda x,y: x+y, [self.ds(b) for b in boundaryMarker] )
        
    def nameRegion(self, name, regionMarker):
        self.dxR[name] = reduce(lambda x,y: x+y, [self.dx(r) for r in regionMarker] )
        
        

def getMesh(meshGMSH,  meshXmlFile, create = True):
    
    radFile = meshXmlFile[:-4] + '.{0}'
    
    if(create):
        meshGeoFile = radFile.format('geo')
        meshMshFile = radFile.format('msh')
        meshGMSH.write(meshGeoFile,'geo')
        os.system('gmsh -2 -algo del2d -format msh2 ' + meshGeoFile)
        
        os.system('dolfin-convert {0} {1}'.format(meshMshFile, meshXmlFile))
        
    return EnrichedMesh(meshXmlFile)


class myGmsh(pygmsh.built_in.Geometry):
    def __init__(self):
        super().__init__()    
        self.mesh = None
        
    def write(self,savefile, opt = 'meshio'):
        if(opt == 'meshio'):
            meshio.write(savefile, self.mesh)
        elif(opt == 'fenics'):
            iofe.exportMeshHDF5_fromGMSH(self.mesh, savefile)
        elif(opt == 'geo'):
            f = open(savefile,'w')
            f.write(self.get_code())
            f.close()
            
    def generate(self , gmsh_opt = ['']):
        self.mesh = pygmsh.generate_mesh(self, extra_gmsh_arguments = gmsh_opt )

class ellipseMesh(myGmsh):
    def __init__(self, ellipseData, Lx, Ly , lcar):
        super().__init__()    

        self.Lx = Lx
        self.Ly = Ly
        self.lcar = lcar    
        self.eList = self.createEllipses(ellipseData,lcar)
        self.createSurfaces()
        self.physicalNaming()

    def createSurfaces(self):
        self.rec = self.add_rectangle(0.0,self.Lx,0.0,self.Ly, 0.0, lcar=self.lcar, holes = self.eList)
    
    def physicalNaming(self):
        self.add_physical(self.rec.surface, 'vol')
        [self.add_physical(e,'ellipse' + str(i)) for i, e in enumerate(self.eList)]
        [self.add_physical(e,'side' + str(i)) for i, e in enumerate(self.rec.lines)]
        
        
    def createEllipses(self, ellipseData, lcar):
        eList = []
        
        angles = [0.0, 0.5*np.pi, np.pi, 1.5*np.pi]
        for cx, cy, l, e, t in ellipseData: # center, major axis length, excentricity, theta
            lenghts = [l,e*l,l,e*l]
            pc = self.add_point([cx,cy,0.0], lcar = lcar)
            pi =  [ self.add_point([cx + li*np.cos(ti + t), cy + li*np.sin(ti + t), 0.0], lcar = lcar) for li, ti in zip(lenghts,angles)]
            ai = [self.add_ellipse_arc(pi[i],pc,pi[i], pi[(i+1)%4]) for i in range(4)] # start, center, major axis, end
            a = self.add_line_loop(lines = ai)
            eList.append(self.add_surface(a))
        
        return eList
    
    def setTransfiniteBoundary(self,n):
        self.set_transfinite_lines(self.rec.lines, n)
                

class ellipseMesh2Domains(ellipseMesh):
    def __init__(self, x0L, y0L, LxL, LyL, NL, ellipseData, Lx, Ly , lcar, x0 = 0., y0 = 0. ):        
        self.x0 = x0
        self.y0 = y0
        self.x0L = x0L
        self.y0L = y0L
        self.LxL = LxL
        self.LyL = LyL
        self.NL = NL
        super().__init__(ellipseData, Lx, Ly , lcar)    
        
    def createSurfaces(self):        
        self.recL = self.add_rectangle(self.x0L, self.x0L + self.LxL, self.y0L , self.y0L + self.LyL, 0.0, lcar=self.lcar, holes = self.eList[:self.NL])                 
        self.rec = self.add_rectangle(self.x0,self.x0 + self.Lx, self.y0, self.y0 + self.Ly, 0.0, lcar=self.lcar, holes = self.eList[self.NL:] + [self.recL])
    
    def physicalNaming(self):
        # self.add_physical(self.recL.surface, 'volL')
        # super().physicalNaming()

        self.add_physical(self.eList[:self.NL],0)    
        self.add_physical(self.recL.surface, 1)
        self.add_physical(self.eList[self.NL:],2)
        self.add_physical(self.rec.surface, 3)
        self.add_physical(self.rec.lines,4)        

    def setTransfiniteInternalBoundary(self,n):
        self.set_transfinite_lines(self.recL.lines, n)

# class ellipseMesh2DomainsPhysicalMeaning(ellipseMesh2Domains):
#     def physicalNaming(self):
#         self.add_physical(self.rec.lines,1)
#         self.add_physical(self.recL.lines,2)
#         self.add_physical(self.recL.surface, 3)
#         self.add_physical(self.eList[:self.NL],4)
#         self.add_physical(self.rec.surface, 5)
#         self.add_physical(self.eList[self.NL:],6)
#         # self.add_physical(self.recL.lines,'boundaryL')



# class ellipseMesh2DomainsPhysicalMeaning(ellipseMesh2Domains):
#     def physicalNaming(self):
#         self.add_physical(self.recL.surface, 0)
#         self.add_physical(self.eList[:self.NL],1)
#         self.add_physical(self.rec.surface, 2)
#         self.add_physical(self.eList[self.NL:],3)
#         self.add_physical(self.rec.lines[0],4)
#         self.add_physical(self.rec.lines[1],5)
#         self.add_physical(self.rec.lines[2],6)
#         self.add_physical(self.rec.lines[3],7)
#         self.add_physical(self.recL.lines,8)

        # self.add_physical(self.recL.lines,'boundaryL')

class ellipseMesh2DomainsPhysicalMeaning(ellipseMesh2Domains):
    def physicalNaming(self):
        self.add_physical(self.recL.surface, 0)
        self.add_physical(self.eList[:self.NL],1)
        self.add_physical(self.rec.surface, 2)
        self.add_physical(self.eList[self.NL:],3)
        self.add_physical(self.rec.lines,4)
        self.add_physical(self.recL.lines,5)



        
class ellipseMeshRepetition(ellipseMesh):
    def __init__(self, times, ellipseData, Lx, Ly , lcar):
        
        n = ellipseData.shape[0]
        ellipseDataNew = np.zeros((times*times*n,5))
        
        fac = 1.0/times
        
        for i in range(times):
            for j in range(times):
                for k in range(n):
                    kk = i*times*n + j*n + k
                    ellipseDataNew[kk,0] = fac*(ellipseData[k,0] + i) # cx
                    ellipseDataNew[kk,1] = fac*(ellipseData[k,1] + j) # cy
                    ellipseDataNew[kk,2] = fac*ellipseData[k,2] # r 
                    ellipseDataNew[kk,3:] = ellipseData[k,3:] 
                    
        super().__init__(ellipseDataNew, Lx, Ly , lcar)    

    
    def physicalNaming(self):
        self.add_physical(self.eList[:],0)
        self.add_physical(self.rec.surface, 1)
        self.add_physical(self.rec.lines,2)        


    def setTransfiniteInternalBoundary(self,n):
        self.set_transfinite_lines(self.recL.lines, n)
        
class ellipseMesh2(myGmsh):
    # def __init__(self, Lx = 1.0,Ly= 1.0, ellipseData = [], lcar = 0.05, ifPeriodic = False):
    def __init__(self, ellipseData, x0, y0, Lx, Ly , lcar):
        super().__init__()    
        
        self.x0 = x0
        self.y0 = y0
        self.Lx = Lx
        self.Ly = Ly
        self.lcar = lcar    
        self.eList = self.createEllipses(ellipseData,lcar)
        self.createSurfaces()
        self.physicalNaming()

    def createSurfaces(self):
        self.rec = self.add_rectangle(self.x0,self.x0 + self.Lx,self.y0,self.y0 + self.Ly, 0.0, lcar=self.lcar, holes = self.eList)
    
    def physicalNaming(self):
        self.add_physical(self.rec.surface, 1)
        self.add_physical(self.eList[:],0)
        self.add_physical(self.rec.lines,2)                
        
    def createEllipses(self, ellipseData, lcar):
        eList = []
        
        angles = [0.0, 0.5*np.pi, np.pi, 1.5*np.pi]
        for cx, cy, l, e, t in ellipseData: # center, major axis length, excentricity, theta
            lenghts = [l,e*l,l,e*l]
            pc = self.add_point([cx,cy,0.0], lcar = lcar)
            pi =  [ self.add_point([cx + li*np.cos(ti + t), cy + li*np.sin(ti + t), 0.0], lcar = lcar) for li, ti in zip(lenghts,angles)]
            ai = [self.add_ellipse_arc(pi[i],pc,pi[i], pi[(i+1)%4]) for i in range(4)] # start, center, major axis, end
            a = self.add_line_loop(lines = ai)
            eList.append(self.add_surface(a))
        
        return eList
    
    def setTransfiniteBoundary(self,n):
        self.set_transfinite_lines(self.rec.lines, n)
                
        
# class ex3D(myGmsh):
#     def __init__(self):
#         super(myGmsh,self).__init__()
        
#         # Draw a cross.
#         poly = self.add_polygon([
#             [ 0.0,  0.5, 0.0],
#             [-0.1,  0.1, 0.0],
#             [-0.5,  0.0, 0.0],
#             [-0.1, -0.1, 0.0],
#             [ 0.0, -0.5, 0.0],
#             [ 0.1, -0.1, 0.0],
#             [ 0.5,  0.0, 0.0],
#             [ 0.1,  0.1, 0.0]
#             ],
#             lcar=0.05
#         )
        
#         axis = [0, 0, 1]
        
#         self.extrude(
#             poly,
#             translation_axis=axis,
#             rotation_axis=axis,
#             point_on_axis=[0, 0, 0],
#             angle=2.0 / 6.0 * np.pi
#         )