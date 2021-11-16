from core.mesh.wrapper_gmsh import myGmsh
from core.mesh.ellipse_mesh import ellipseMesh
import numpy as np

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