import numpy as np
import meshio

import pygmsh

import ioFenicsWrappers as iofe


class myGmsh(object):
    def __init__(self, mesh):
        self.mesh = mesh
        
    def write(self,savefile, opt = 'meshio'):
        if(opt == 'meshio'):
            meshio.write(savefile, self.mesh_gmsh)
        elif(opt == 'fenics'):
            iofe.exportMeshHDF5_fromGMSH(self.mesh, savefile)
        elif(opt == 'geo'):
            f = open(savefile,'w')
            f.write(self.mesh.get_code())
            f.close()
            
    def generate(self , gmsh_opt = ['']):
        self.mesh_gmsh = pygmsh.generate_mesh(self.mesh, extra_gmsh_arguments = gmsh_opt )
        

class ellipseMesh(pygmsh.built_in.Geometry, object):
    # def __init__(self, Lx = 1.0,Ly= 1.0, ellipseData = [], lcar = 0.05, ifPeriodic = False):
    def __init__(self, ellipseData, Lx, Ly , lcar, ifPeriodic):
        super().__init__()    

        self.Lx = Lx
        self.Ly = Ly
        self.lcar = lcar
        self.ifPeriodic = ifPeriodic        
        self.eList = self.createEllipses(ellipseData,lcar)
        

    def createSurfaces(self):
        rec = self.add_rectangle(0.0,self.Lx,0.0,self.Ly, 0.0, lcar=self.lcar, holes = self.eList)
    
        [self.add_physical(e,'side' + str(i)) for i, e in enumerate(rec.lines)]
        self.add_physical(rec.surface, 'vol')
        [self.add_physical(e,'ellipse' + str(i)) for i, e in enumerate(self.eList)]
        
    def createEllipses(self, ellipseData, lcar):
        eList = []
        
        for cx, cy, l, e, t in ellipseData: # center, major axis length, excentricity, theta
            pc = self.add_point([cx,cy,0.0], lcar = lcar)
            pi =  [ self.add_point([cx + li*np.cos(ti), cy + li*np.sin(ti), 0.0], lcar = lcar) for li, ti in zip([l,e*l,l,e*l], np.linspace(0.0,1.5*np.pi,4) + t) ]
            ai = [self.add_ellipse_arc(pi[i],pc,pi[i], pi[(i+1)%4]) for i in range(4)] # start, center, major axis, end
            a = self.add_line_loop(lines = ai)
            eList.append(self.add_surface(a))
        
        return eList

        # if(ifPeriodic):
        #     self.add_raw_code("Periodic Curve {{{}}} = {{{}}};".format(rec.lines[0].id, rec.lines[2].id))
        #     self.add_raw_code("Periodic Curve {{{}}} = {{{}}};".format(rec.lines[1].id, rec.lines[3].id))

class ellipseMesh2Domains(pygmsh.built_in.Geometry, object):
    # def __init__(self, x0L, y0L, LxL, LyL , NL, ellipseData = [], Lx = 1.0 , Ly= 1.0, lcar = 0.05, ifPeriodic = False):       
    def __init__(self, x0L, y0L, LxL, LyL , NL, ellipseData, Lx, Ly, lcar, ifPeriodic):
        super().__init__()    

        self.Lx = Lx
        self.Ly = Ly
        self.lcar = lcar
        self.ifPeriodic = ifPeriodic        
        self.x0L = x0L
        self.y0L = y0L
        self.LxL = LxL
        self.LyL = LyL
        self.NL = NL
        self.ellipseData = ellipseData
    
        self.eList = self.createEllipses(ellipseData,lcar)
    
        recL = self.add_rectangle(self.x0L, self.x0L + self.LxL, self.y0L , self.y0L + self.LyL, 0.0, lcar=self.lcar, holes = self.eList[:self.NL])                 
        rec = self.add_rectangle(0.0,self.Lx,0.0,self.Ly, 0.0, lcar=self.lcar, holes = self.eList[self.NL:] + [recL])
    
        [self.add_physical(e,'side' + str(i)) for i, e in enumerate(rec.lines)]
        self.add_physical(recL.surface, 'volL')
        self.add_physical(rec.surface, 'vol')
        [self.add_physical(e,'ellipse' + str(i)) for i, e in enumerate(self.eList)]
        
    def createEllipses(self, ellipseData, lcar):
        eList = []
        
        for cx, cy, l, e, t in ellipseData: # center, major axis length, excentricity, theta
            pc = self.add_point([cx,cy,0.0], lcar = lcar)
            pi =  [ self.add_point([cx + li*np.cos(ti), cy + li*np.sin(ti), 0.0], lcar = lcar) for li, ti in zip([l,e*l,l,e*l], np.linspace(0.0,1.5*np.pi,4) + t) ]
            ai = [self.add_ellipse_arc(pi[i],pc,pi[i], pi[(i+1)%4]) for i in range(4)] # start, center, major axis, end
            a = self.add_line_loop(lines = ai)
            eList.append(self.add_surface(a))
        
        return eList
    
    # def createSurfaces(self):        
    #     recL = self.add_rectangle(self.x0L, self.x0L + self.LxL, self.y0L , self.y0L + self.LyL, 0.0, lcar=self.lcar, holes = self.eList[:self.NL])                 
    #     rec = self.add_rectangle(0.0,self.Lx,0.0,self.Ly, 0.0, lcar=self.lcar, holes = self.eList[self.NL:] + [recL])
    
    #     [self.add_physical(e,'side' + str(i)) for i, e in enumerate(rec.lines)]
    #     self.add_physical(recL.surface, 'volL')
    #     self.add_physical(rec.surface, 'vol')
    #     [self.add_physical(e,'ellipse' + str(i)) for i, e in enumerate(self.eList)]

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
