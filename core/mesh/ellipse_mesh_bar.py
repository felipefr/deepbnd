from core.mesh.ellipse2_mesh import ellipseMesh2
        
class ellipseMeshBar(ellipseMesh2):
    
    def physicalNaming(self):
        self.add_physical(self.rec.surface, 1)
        self.add_physical(self.eList[:],0)
        [self.add_physical(self.rec.lines[i],2+i) for i in range(4)]  #bottom, right, top, left
    
