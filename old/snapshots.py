import numpy as np
import os
import ioFenicsWrappers as iofe
import fenicsWrapperElasticity as fela
from dolfin import HDF5File, MPI, Function, FunctionSpace
import h5py 
import myCoeffClass as coef
from pathlib import Path
import copy

def getMapParamToDomain(p, paramAux = None):
    if(type(paramAux) == type(None)):
        paramAux = np.zeros((10,2))
        
    paramAux[0:9,0] = p[0] ; paramAux[0:9,1] = p[2]
    paramAux[9,0] = p[1]; paramAux[9,1] = p[3]
    
    return paramAux
    
def getMapParamToDomain2(p, paramAux = None):
    if(type(paramAux) == type(None)):
        paramAux = np.zeros((3,2))
        
    paramAux[0,0] = p[0] ; paramAux[0,1] = p[2]
    paramAux[1:3,0] = p[1] ; paramAux[1:3,1] = p[3]
    
    return paramAux 

class Snapshots:
    
    def __init__(self, radical, femData, mode = 'safe'):

        self.mode = mode
        self.radical = radical        
        self.meshFile = self.radical + 'mesh.xdmf'
        self.solutionFile = self.radical + 'solution.hdf5'
        self.posProcFile = self.radical + 'posProc.hdf5'
        self.paramFile = self.radical + 'param.hdf5'
        self.datasetFile = self.radical + 'dataset.hdf5'

        self.params = {}
        
        self.femData = femData
        self.femProblem = self.femData['problem']
        self.defaultMeshParam = self.femData['defaultMeshParam']
        
        self.defaultCompression = {'dtype' : 'f8',  'compression' : "gzip", 
                                   'compression_opts' : 1, 'shuffle' : True}
        
        self.Mesh = None
        self.MeshRef = None
        
        self.getMapParamToDomain = getMapParamToDomain
        
    def checkfiles(self):
        files = [self.meshFile, self.solutionFile, self.posProcFile, self.paramFile, self.datasetFile]
        existence = []
        
        print('Summary Files')
        for f in files:
            existence.append(Path(f).is_file())
            print(f + ' exists? ={0}'.format(existence[-1]))
            
        assert existence[0], "cannot proceed without a mesh"
        
        if(self.mode == 'copy'):
            for e, f in zip(existence[1:], files[1:]):
                if(e):
                    fold = copy.deepcopy(f)
                    f = f[:-5] + "_copy_" + f[-5:]
                    os.system("cp " + fold + " " + f)
                    
        elif(self.mode == 'removal'):
            for e, f in zip(existence[1:], files[1:]):
                if(e):
                    os.system("rm " + f)
                    
        elif(self.mode == 'ignore'):
            pass
        else:
            assert not np.any(np.array(existence[1:])), "Any of the other files already exists, run in copy or removal mode"
                  
                
        
    def loadMeshes(self, meshFile = None):
        if(meshFile):
            self.meshFile = meshFile
        
        self.MeshRef = fela.EnrichedMesh(self.meshFile)
        self.MeshRef.createFiniteSpace(**self.femData['fespace'])
        self.Mesh = fela.EnrichedMesh(self.meshFile) # try to really create a copy afterwards
        self.Mesh.createFiniteSpace(**self.femData['fespace'])
        
    def registerParameters(self, p):
        for pi in p:
            self.params[pi.name] = pi 
        
    def writeBasics(self): 
        u = Function(self.Mesh.V['u']) 
        self.Nh = u.vector().get_local().size
        
        with HDF5File(MPI.comm_world, self.solutionFile, 'w') as f:     
            f.write(u, "basic")
        
        del u

    def buildSnapshots(self,label = 'Train', indexes = []):
        
        self.checkfiles()
        self.loadMeshes()
        self.writeBasics()
        
        if(len(indexes) == 0):
            ns = self.params["pm"].samples[label].shape[0]
            indexes = np.arange(ns).astype('int')
        else:
            ns = len(indexes)
            
        with h5py.File(self.solutionFile, 'a') as f:    
            sol = f.create_dataset(label + '/sol', shape =  (self.Nh, ns), **self.defaultCompression)
            # sol = f.create_dataset(label + '/sol', shape =  (self.Nh, self.ns), dtype='f8')
            
            paramAux = self.getMapParamToDomain(4*[0]) # just to initialize of the right size
                    
            for i in indexes:
                print("building snapshot {0} {1}".format(label,i))
    
                self.getMapParamToDomain(self.params["pd"](i,label),paramAux)
                
                iofe.moveMesh(self.Mesh, self.MeshRef, self.params["pm"](i,label,['tx','ty','lx','ly']), 
                              self.defaultMeshParam)
                sol[:,i] = self.femProblem(paramAux, self.Mesh)

               
        
    def posProcessingStress(self, label, indexes = []):
        if(indexes):
            ns = len(indexes)
            ib = indexes[0]
        else:
            ns = self.params["pm"].samples[label].shape[0]
            indexes = np.arange(ns).astype('int')
            ib = 0
            
        u = self.load_basicStructureU() 
  
        with h5py.File(self.solutionFile, 'r') as f, h5py.File(self.posProcFile, 'w') as g:
            solU = f[label + '/sol']
            vonMisesSol = g.create_dataset(label + '/vonMises', shape =  (ns,self.Mesh.cells().shape[0]), 
                                           **self.defaultCompression)
            
            
            materials = self.Mesh.subdomains.array().astype('int32') - np.max(self.Mesh.boundaries.array().astype('int32')) - 1
            paramAux = self.getMapParamToDomain(4*[0]) # just to initialize of the right size
            lame = coef.getMyCoeff(materials , paramAux, op = 'cpp') 
            sigma = lambda u: fela.sigmaLame(u,lame)
            
            Vsig = FunctionSpace(self.Mesh, "DG", 0)
            von_Mises = Function(Vsig, name="Stress")
                    
            for i in indexes:
                print("posprocessing {0} {1}".format(label,i))
                
                self.getMapParamToDomain(self.params["pd"](i,label),paramAux)
                lame.updateCoeffs(paramAux)
                
                iofe.moveMesh(self.Mesh, self.MeshRef, 
                              self.params["pm"](i,label,['tx','ty','lx','ly']), self.defaultMeshParam)
                
                u.vector().set_local(solU[:,i])
                von_Mises_ = fela.vonMises(sigma(u))
     
                von_Mises.assign(iofe.local_project(von_Mises_, Vsig))
                vonMisesSol[i-ib,:] = von_Mises.vector().get_local()


    def exportVTK(self, outputFile, label, indexes):
        if(not self.Mesh):
            self.loadMeshes()
        
        u = self.load_basicStructureU()
        paramAux = self.getMapParamToDomain(4*[0]) # just to initialize of the right size
        
        with h5py.File(self.solutionFile, 'r') as f:
            solU = f[label + '/sol']
            for i in indexes:
                self.getMapParamToDomain(self.params["pd"](i,label), paramAux)
                iofe.moveMesh(self.Mesh, self.MeshRef, 
                        self.params["pm"](i,label,['tx','ty','lx','ly']), self.defaultMeshParam)
                        
                u.vector().set_local(solU[:,i])
                outputFileNew = "{0}_{2}.{1}".format(*outputFile.split('.'),i)
                iofe.postProcessing_complete(u, outputFileNew, labels = ['u','vonMises','lame'], param = paramAux)

    def load_basicStructureU(self):
        u = Function(self.Mesh.V['u'])
        with HDF5File(MPI.comm_world, self.solutionFile, 'r') as f:
            f.read(u, 'basic')
           
        return u
    
    def generateData(self, generators, mode = 'w'):
        
        if(not self.Mesh):
            self.loadMeshes()

        with h5py.File(self.datasetFile, mode) as f, \
             h5py.File(self.solutionFile, 'r') as f0, \
             h5py.File(self.posProcFile, 'r') as f1:
                 
            for g in generators:
                requiredFields = []
                for l in g.requiredFields():
                    
                    if(l == 'basic'):
                        requiredFields.append(self.load_basicStructureU())
                    else: 
                        requiredFields.append(g.filechoice([f0,f1])[g.labelSample + '/' + l ])
                                           
                data = f.create_dataset(g.labelSample + '/' + g.label, shape = g.shape(), **self.defaultCompression)
                g(data, self.Mesh, self.MeshRef, self.params["pm"], requiredFields, self.defaultMeshParam)
                
                f.flush()
        
        