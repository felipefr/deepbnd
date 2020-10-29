import pygmsh
import sys, os
from numpy import isclose
sys.path.insert(0, '../utils/')
import matplotlib.pyplot as plt
import numpy as np
import generationInclusions as geni
import gmsh

with pygmsh.geo.Geometry() as geom:
    lcar = 0.1
    print(type(geom))
    p1 = geom.add_point([0.0, 0.0], lcar)
    p2 = geom.add_point([1.0, 0.0], lcar)
    p3 = geom.add_point([1.0, 0.5], lcar)
    p4 = geom.add_point([1.0, 1.0], lcar)
    s1 = geom.add_bspline([p1, p2, p3, p4])

    p2 = geom.add_point([0.0, 1.0], lcar)
    p3 = geom.add_point([0.5, 1.0], lcar)
    s2 = geom.add_spline([p4, p3, p2, p1])

    ll = geom.add_curve_loop([s1, s2])
    pl = geom.add_plane_surface(ll)

    mesh = geom.generate_mesh()


class myGmsh(pygmsh.geo.Geometry): 
    def __init__(self):
        super(myGmsh,self).__init__()
        self.mesh = None
        self.radFileMesh = 'defaultMesh.{0}'
        self.format = 'xdmf'
        # self.geo = pygmsh.geo.Geometry()
class ellipseMesh2(myGmsh):
    # def __init__(self, Lx = 1.0,Ly= 1.0, ellipseData = [], lcar = 0.05, ifPeriodic = False):
    def __init__(self, ellipseData, x0, y0, Lx, Ly , lcar):
        super(ellipseMesh2,self).__init__()    
        
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
        print(lcar)
        print(ellipseData)
        print(self.env)

        
        angles = [0.0, 0.5*np.pi, np.pi, 1.5*np.pi]
        for cx, cy, l, e, t in ellipseData: # center, major axis length, excentricity, theta
            lenghts = [l,e*l,l,e*l]
            pc = self.add_point([cx,cy], self.lcar)
            pi =  [ self.add_point([cx + li*np.cos(ti + t), cy + li*np.sin(ti + t)], self.lcar) for li, ti in zip(lenghts,angles)]
            ai = [self.add_ellipse_arc(pi[i],pc,pi[i], pi[(i+1)%4]) for i in range(4)] # start, center, major axis, end
            a = self.add_line_loop(lines = ai)
            eList.append(self.add_surface(a))
        
        return eList
    
    def setTransfiniteBoundary(self,n):
        self.set_transfinite_lines(self.rec.lines, n)
                


ellipseData, PermTotal, PermBox = geni.circularRegular2Regions(r0 = 0.2, r1 = 0.3, NxL =2, NyL=2, Lx=1.0, Ly = 1.0, offset = 0, ordered = False)

with ellipseMesh2(ellipseData,0,0,1,1,0.1) as m:
    m.setTransfiniteBoundary(10)

