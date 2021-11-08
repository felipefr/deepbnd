import numpy as np
import meshio
import pygmsh
import ioFenicsWrappers as iofe
import os
import dolfin as df
from functools import reduce

class myGmsh(pygmsh.built_in.Geometry):
    def __init__(self):
        super().__init__()    
        self.mesh = None
        self.radFileMesh = 'defaultMesh.{0}'
        self.format = 'xdmf'
        
    def writeGeo(self, savefile = ''):
        if(len(savefile) == 0):
            savefile = self.radFileMesh.format('geo')
            
        f = open(savefile,'w')
        f.write(self.get_code())
        f.close()

    def writeXML(self, savefile = ''):
        if(len(savefile) == 0):
            savefile = self.radFileMesh.format('xml')
        else:
            self.radFileMesh = savefile[:-4] + '.{0}'
         
        meshGeoFile = self.radFileMesh.format('geo')
        meshMshFile = self.radFileMesh.format('msh')
        self.writeGeo(meshGeoFile)
        os.system('gmsh -2 -format msh2 -algo del2d' + meshGeoFile) # with del2d, noticed less distortions
        os.system('dolfin-convert {0} {1}'.format(meshMshFile, savefile))    
    
    def write(self,savefile = '', opt = 'meshio'):
        if(type(self.mesh) == type(None)):
            # self.generate(gmsh_opt=['-bin','-v','1', '-algo', 'del2d']) # with del2d, noticed less distortions      
            self.generate(gmsh_opt=['-bin','-v','1', '-algo', 'front2d', 
                                    '-smooth', '2',  '-anisoMax', '1000.0']) # with del2d, noticed less distortions      
        if(len(savefile) == 0):
            savefile = self.radFileMesh.format('xdmf')
        
        if(opt == 'meshio'):
            meshio.write(savefile, self.mesh)
        elif(opt == 'fenics'):
            iofe.exportMeshHDF5_fromGMSH(self.mesh, savefile)
        
        # return self.mesh

            
    def generate(self , gmsh_opt = ['']):
        self.mesh = pygmsh.generate_mesh(self, extra_gmsh_arguments = gmsh_opt, dim = 2,mesh_file_type = 'msh2') # it should be msh2 cause of tags    
        # self.mesh = pygmsh.generate_mesh(self, verbose=False, dim=2, prune_vertices=True, prune_z_0=True,
                                          # remove_faces=False, extra_gmsh_arguments=gmsh_opt,  mesh_file_type='msh4') # it should be msh2 cause of tags

    def getEnrichedMesh(self, savefile = ''):
        
        if(len(savefile) == 0):
            savefile = self.radFileMesh.format(self.format)
        
        if(savefile[-3:] == 'xml'):
            self.writeXML(savefile)
            
        elif(savefile[-4:] == 'xdmf'):
            print("exporting to fenics")
            self.write(savefile, 'fenics')
        
        return EnrichedMesh(savefile)
    
    def setNameMesh(self,nameMesh):
        nameMeshSplit = nameMesh.split('.')
        self.format = nameMeshSplit[-1]
        self.radFileMesh = reduce(lambda x,y : x + '.' + y, nameMeshSplit[:-1])
        self.radFileMesh += ".{0}"
        


class rectangleMesh(myGmsh):
    def __init__(self, x0, y0, Lx, Ly , lcar):
        super().__init__()
        
        self.x0 = x0
        self.y0 = y0
        self.Lx = Lx
        self.Ly = Ly
        self.lcar = lcar    
        self.createSurfaces()
        self.physicalNaming()

    def createSurfaces(self):
        self.rec = self.add_rectangle(self.x0,self.x0 + self.Lx,self.y0,self.y0 + self.Ly, 0.0, lcar=self.lcar)
    
    def physicalNaming(self):
        self.add_physical(self.rec.surface, 'vol')
        [self.add_physical(e,'side' + str(i)) for i, e in enumerate(self.rec.lines)]
    
    def setTransfiniteBoundary(self,n):
        self.set_transfinite_lines(self.rec.lines, n)
        
class degeneratedBoundaryRectangleMesh(myGmsh): # Used for implementation of L2bnd
    def __init__(self, x0, y0, Lx, Ly, Nb): 
        super().__init__()
        
        self.x0 = x0
        self.y0 = y0
        self.Lx = Lx
        self.Ly = Ly
        self.lcar = 2*self.Lx    ## just a huge value
        self.createSurfaces()
        self.physicalNaming()
        self.set_transfinite_lines(self.extLines, Nb)

    def createSurfaces(self):
        p1 = self.add_point([self.x0, self.y0, 0.0], lcar = self.lcar)
        p2 = self.add_point([self.x0 + self.Lx, self.y0, 0.0], lcar = self.lcar)
        p3 = self.add_point([self.x0 + self.Lx, self.y0 + self.Ly ,0.0], lcar = self.lcar)
        p4 = self.add_point([self.x0 , self.y0 + self.Ly ,0.0], lcar = self.lcar)
        p5 = self.add_point([self.x0 + 0.5*self.Lx, self.y0 + 0.5*self.Ly ,0.0], lcar = self.lcar)
        
        p = [p1,p2,p3,p4]
        
        self.extLines = [ self.add_line(p[i],p[(i+1)%4]) for i in range(4) ]
        self.intLines = [ self.add_line(p[i],p5) for i in range(4) ]
        
        LineLoops = [ self.add_line_loop(lines = [-self.intLines[i], self.extLines[i] ,self.intLines[(i+1)%4]]) for i in range(4)]
        self.Surfs = []
        for ll in LineLoops:
            self.Surfs.append(self.add_surface(ll))
    
    def physicalNaming(self):
        self.add_physical(self.Surfs, 'vol')
        [self.add_physical(e,'side' + str(i)) for i, e in enumerate(self.extLines)]
    

class ellipseMesh(myGmsh):
    def __init__(self, ellipseData, Lx, Ly , lcar):
        super().__init__()  
        
        self.lcar = lcar   

        if(type(self.lcar) is not type([])):
            self.lcar = len(ellipseData)*[self.lcar]
            
        self.Lx = Lx
        self.Ly = Ly
 
        self.eList = self.createEllipses(ellipseData,self.lcar)
        self.createSurfaces()
        self.physicalNaming()

    def createSurfaces(self):
        self.rec = self.add_rectangle(0.0,self.Lx,0.0,self.Ly, 0.0, lcar=self.lcar[-1], holes = self.eList)
    
    def physicalNaming(self):
        self.add_physical(self.rec.surface, 'vol')
        [self.add_physical(e,'ellipse' + str(i)) for i, e in enumerate(self.eList)]
        [self.add_physical(e,'side' + str(i)) for i, e in enumerate(self.rec.lines)]
        
        
    def createEllipses(self, ellipseData, lcar):
        eList = []
            
        ilcar_current = 0
        angles = [0.0, 0.5*np.pi, np.pi, 1.5*np.pi]
        for cx, cy, l, e, t in ellipseData: # center, major axis length, excentricity, theta
            lenghts = [l,e*l,l,e*l]
            pc = self.add_point([cx,cy,0.0], lcar = lcar[ilcar_current])
            pi =  [ self.add_point([cx + li*np.cos(ti + t), cy + li*np.sin(ti + t), 0.0], lcar = lcar[ilcar_current]) for li, ti in zip(lenghts,angles)]
            ai = [self.add_ellipse_arc(pi[i],pc,pi[i], pi[(i+1)%4]) for i in range(4)] # start, center, major axis, end
            a = self.add_line_loop(lines = ai)
            eList.append(self.add_surface(a))
            ilcar_current+=1
        
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
        self.recL = self.add_rectangle(self.x0L, self.x0L + self.LxL, self.y0L , self.y0L + self.LyL, 0.0, lcar=self.lcar[0], holes = self.eList[:self.NL])                 
        self.rec = self.add_rectangle(self.x0,self.x0 + self.Lx, self.y0, self.y0 + self.Ly, 0.0, lcar=self.lcar[-1], holes = self.eList[self.NL:] + [self.recL])
    
    def physicalNaming(self):
        # self.add_physical(self.recL.surface, 'volL''
        # super().physicalNaming()

        self.add_physical(self.eList[:self.NL],0)    
        self.add_physical(self.recL.surface, 1)
        self.add_physical(self.eList[self.NL:],2)
        self.add_physical(self.rec.surface, 3)
        self.add_physical(self.rec.lines,4)        

    def setTransfiniteInternalBoundary(self,n):
        self.set_transfinite_lines(self.recL.lines, n)

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
        
        
class ellipseMeshBar(ellipseMesh2):
    
    def physicalNaming(self):
        self.add_physical(self.rec.surface, 1)
        self.add_physical(self.eList[:],0)
        [self.add_physical(self.rec.lines[i],2+i) for i in range(4)]  #bottom, right, top, left
    

            
class ellipseMeshBarAdaptative(myGmsh):
    def __init__(self, ellipseData, x0, y0, Lx, Ly , lcar): # lcar[1]<lcar[0]<lcar[2]
        super().__init__()    
        
        self.x0 = x0
        self.y0 = y0
        self.Lx = Lx
        self.Ly = Ly
        self.lcar = lcar    
        self.eList = self.createEllipses(ellipseData)
        self.createSurfaces()
        self.physicalNaming()

    def createSurfaces(self):
        self.rec = self.add_rectangle(self.x0,self.x0 + self.Lx,self.y0,self.y0 + self.Ly, 0.0, lcar=self.lcar[0], holes = self.eList)
    
    def physicalNaming(self):
        self.add_physical(self.rec.surface, 1)
        self.add_physical(self.eList[:],0)
        [self.add_physical(self.rec.lines[i],2+i) for i in range(4)]  #bottom, right, top, left
        
    def createEllipses(self, ellipseData):
        eList = []
        
        lcar = self.lcar
        angles = [0.0, 0.5*np.pi, np.pi, 1.5*np.pi]
        for cx, cy, l, e, t in ellipseData: # center, major axis length, excentricity, theta
            lenghts = [l,e*l,l,e*l]
            pc = self.add_point([cx,cy,0.0], lcar = lcar[2])
            pi =  [ self.add_point([cx + li*np.cos(ti + t), cy + li*np.sin(ti + t), 0.0], lcar = lcar[1]) for li, ti in zip(lenghts,angles)]
            ai = [self.add_ellipse_arc(pi[i],pc,pi[i], pi[(i+1)%4]) for i in range(4)] # start, center, major axis, end
            a = self.add_line_loop(lines = ai)
            eList.append(self.add_surface(a))
            
        return eList
        
    def setTransfiniteBoundary(self,n, direction = 'horiz'):
        if(direction == 'horiz'):
            self.set_transfinite_lines(self.rec.lines[0::2], n)
        else:
            self.set_transfinite_lines(self.rec.lines[1::2], n)
                


class ellipseMeshBarAdaptative_3circles(myGmsh):
    def __init__(self, ellipseData, x0, y0, Lx, Ly , lcar, he): # lcar[0]<lcar[1]<lcar[2]
        super().__init__()    
        
        self.x0 = x0
        self.y0 = y0
        self.Lx = Lx
        self.Ly = Ly
        self.lcar = lcar  
        self.he = he
        self.eList0, self.eList1, self.eList2 = self.createEllipses(ellipseData)
        self.createSurfaces()
        self.physicalNaming()

    def createSurfaces(self):
        self.rec = self.add_rectangle(self.x0,self.x0 + self.Lx,self.y0,self.y0 + self.Ly, 0.0, lcar=self.lcar[0], holes = self.eList2)
    
    def physicalNaming(self):
        self.add_physical(self.eList0[:] + self.eList1[:],0)
        self.add_physical(self.eList2[:] + [self.rec.surface], 1)
        [self.add_physical(self.rec.lines[i],2+i) for i in range(4)]  #bottom, right, top, left
        
    def createEllipses(self, ellipseData):
        eList0 = []
        eList1 = [] 
        eList2 = []
        
        lcar = self.lcar
        angles = [0.0, 0.5*np.pi, np.pi, 1.5*np.pi]
        
        he0, he1 = self.he
        pc = []

        for cx, cy, l, e, t in ellipseData: # center, major axis length, excentricity, theta
            pc.append(self.add_point([cx,cy,0.0], lcar = 0.5*l))
        
        k = 0
        for cx, cy, l, e, t in ellipseData: # center, major axis length, excentricity, theta
            lenghts = [l,e*l,l,e*l]
            lenghts = [li - he0 for li in lenghts]
            pi =  [ self.add_point([cx + li*np.cos(ti + t), cy + li*np.sin(ti + t), 0.0], lcar = lcar[2]) for li, ti in zip(lenghts,angles)]
            ai = [self.add_ellipse_arc(pi[i],pc[k],pi[i], pi[(i+1)%4]) for i in range(4)] # start, center, major axis, end
            a = self.add_line_loop(lines = ai)
            eList0.append(self.add_surface(a))
            k = k + 1
        
        k = 0
        for cx, cy, l, e, t in ellipseData: # center, major axis length, excentricity, theta
            lenghts = [l,e*l,l,e*l]
            pi =  [ self.add_point([cx + li*np.cos(ti + t), cy + li*np.sin(ti + t), 0.0], lcar = lcar[1]) for li, ti in zip(lenghts,angles)]
            ai = [self.add_ellipse_arc(pi[i],pc[k],pi[i], pi[(i+1)%4]) for i in range(4)] # start, center, major axis, end
            a = self.add_line_loop(lines = ai)
            eList1.append(self.add_plane_surface(a, holes = [eList0[k]]))
            k = k + 1


        k = 0
        for cx, cy, l, e, t in ellipseData: # center, major axis length, excentricity, theta
            lenghts = [l,e*l,l,e*l]
            lenghts = [ li + he1 for li in lenghts]
            pi =  [ self.add_point([cx + li*np.cos(ti + t), cy + li*np.sin(ti + t), 0.0], lcar = lcar[0]) for li, ti in zip(lenghts,angles)]
            ai = [self.add_ellipse_arc(pi[i],pc[k],pi[i], pi[(i+1)%4]) for i in range(4)] # start, center, major axis, end
            a = self.add_line_loop(lines = ai)
            eList2.append(self.add_plane_surface(a, holes = [eList1[k]]))
            k = k + 1

        
        print(len(eList0),len(eList1),len(eList2))    
        return eList0, eList1, eList2



        
    def setTransfiniteBoundary(self,n, direction = 'horiz'):
        if(direction == 'horiz'):
            self.set_transfinite_lines(self.rec.lines[0::2], n)
        else:
            self.set_transfinite_lines(self.rec.lines[1::2], n)
                
            
    # def addMeshConstraints(self):
    #     field0 = self.add_boundary_layer(
    #         edges_list=[self.eList[0].line_loop.lines[0]],
    #         hfar=0.1,
    #         hwall_n=0.01,
    #         # hwall_t=0.01,
    #         ratio=1.1,
    #         thickness=0.2,
    #         # anisomax=100.0
    #         )
     
    #     # field1 = geom.add_boundary_layer(
    #     #     nodes_list=[p2],
    #     #     hfar=0.1,
    #     #     hwall_n=0.01,
    #     #     hwall_t=0.01,
    #     #     ratio=1.1,
    #     #     thickness=0.2,
    #     #     anisomax=100.0
    #     #     )
     
    #     self.add_background_field([field0])