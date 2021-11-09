from my_gmsh import myGmsh
from ellipse_mesh import ellipseMesh
from ellipse2_mesh import ellipseMesh2

# class ellipseMesh2DomainsPhysicalMeaning(ellipseMesh2Domains):
#     def physicalNaming(self):
#         self.add_physical(self.recL.surface, 0)
#         self.add_physical(self.eList[:self.NL],1)
#         self.add_physical(self.rec.surface, 2)
#         self.add_physical(self.eList[self.NL:],3)
#         self.add_physical(self.rec.lines,4)
#         self.add_physical(self.recL.lines,5)
        
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