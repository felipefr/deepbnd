from core.mesh.wrapper_gmsh import myGmsh
import numpy as np

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