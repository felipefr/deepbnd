import numpy as np
import meshio
import pygmsh
import os
import dolfin as df
from functools import reduce

from deepBND.core.fenics_tools.wrapper_io import exportMeshHDF5_fromGMSH
from deepBND.core.fenics_tools.enriched_mesh import EnrichedMesh

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
                                    '-smooth', '2',  '-anisoMax', '1000.0', '-nt', '1']) # with del2d, noticed less distortions      
        if(len(savefile) == 0):
            savefile = self.radFileMesh.format('xdmf')
        
        if(opt == 'meshio'):
            meshio.write(savefile, self.mesh)
        elif(opt == 'fenics'):
            exportMeshHDF5_fromGMSH(self.mesh, savefile)
        
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