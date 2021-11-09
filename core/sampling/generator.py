import numpy as np
import ioFenicsWrappers as iofe
import dolfin as df

class dataGenerator:
    def __init__(self, label, labelSample, ns_start, ns_end):
        self.label = label
        self.labelSample = labelSample
        self.ns_start = ns_start
        self.ns_end = ns_end
        self.ns = self.ns_end - self.ns_start
    
    def __call__(self, data, Mesh, MeshRef, pm, RequiredFields, defaultMeshParam):
        pass
    
    def requiredFields(self):
        pass

class stressGenerator(dataGenerator):
    def __init__(self, label, labelSample, Nmax, Nmin, ns_start, ns_end):
        super(stressGenerator, self).__init__(label, labelSample, ns_start, ns_end)
        self.Nmax = Nmax
        self.Nmin = Nmin
 
    
    def __call__(self, data, Mesh, MeshRef, pm, RequiredFields, defaultMeshParam):
        
        vonMisesSol = RequiredFields[0] 
        
        # data = np.zeros(self.shape())
        for i in range(self.ns_start, self.ns_end):
            print("Generating data snapshot stress {0}.{1}".format(self.labelSample,i))
            
            iofe.moveMesh(Mesh, MeshRef, pm(i,self.labelSample,['tx','ty','lx','ly']), defaultMeshParam)
            
            iMinMax = np.argsort(vonMisesSol[i,:])[ np.concatenate((np.arange(self.Nmin),-np.arange(self.Nmax,0,-1))) ] # note that I transposed in the generation
            cells = Mesh.cells()[iMinMax]
            
            for j in range(self.Nmin + self.Nmax):
                data[i-self.ns_start,3*j] = vonMisesSol[i, iMinMax[j]] # note that I transposed in the generation
                data[i-self.ns_start,3*j+1] = np.mean( Mesh.coordinates()[cells[j],0] )
                data[i-self.ns_start,3*j+2] = np.mean( Mesh.coordinates()[cells[j],1] )
        
        # return data
    
    def requiredFields(self):
        return ['vonMises']
    
    def filechoice(self, f): # posprocfile
        return f[1]
    
    def shape(self):
        return (self.ns,3*(self.Nmin + self.Nmax))
    
class displacementGenerator(dataGenerator):
    def __init__(self, label, labelSample, ns_start, ns_end):
        super(displacementGenerator, self).__init__(label, labelSample, ns_start, ns_end)
        
        self.npoints = 0
        self.x_eval = np.zeros((self.npoints,2))
        
    
    def __call__(self, data, Mesh, MeshRef, pm, RequiredFields, defaultMeshParam):       
        u, solU = RequiredFields   
      
        # data = np.zeros(self.shape())
        
        for i in range(self.ns_start, self.ns_end):
            print("Generating data snapshot displacement {0}.{1}".format(self.labelSample,i))
            
            iofe.moveMesh(Mesh, MeshRef, pm(i,self.labelSample,['tx','ty','lx','ly']), defaultMeshParam)
            
            boundTree = Mesh.bounding_box_tree().build(Mesh) # force reconstruction
            
            u.vector().set_local(solU[:,i])
           
            for j in range(self.npoints):
                data[i-self.ns_start,2*j:2*(j+1)] = u(self.x_eval[j,:])
            
            del boundTree
        # return data
    
    def requiredFields(self):
        return ['basic','sol']

    def filechoice(self, f): # solutionfile
        return f[0]
    
    def shape(self):
        return (self.ns,2*self.npoints)


class displacementGeneratorBoundary(displacementGenerator):
    def __init__(self, label, labelSample, Faces, NpointsFace, eps, ns_start, ns_end):
        super(displacementGeneratorBoundary, self).__init__(label, labelSample, ns_start, ns_end)
        
        self.npoints = sum(NpointsFace)
        self.x_eval = np.zeros((self.npoints,2))
        
        k = 0
        for l,n in zip(Faces,NpointsFace):
            if(l=='Right'):
                self.x_eval[k:k+n,0] = 1.0 - eps
                self.x_eval[k:k+n,1] = np.linspace(eps,1.0-eps,n)
            elif(l=='Top'):
                self.x_eval[k:k+n,1] = 1.0 - eps
                self.x_eval[k:k+n,0] = np.linspace(eps,1.0-eps,n)
            elif(l=='Bottom'):
                self.x_eval[k:k+n,1] = eps
                self.x_eval[k:k+n,0] = np.linspace(eps,1.0-eps,n)
                
            k+=n
        

class displacementGeneratorInterior(displacementGenerator):
    def __init__(self, label, labelSample, Nx, Ny, x0, x1, y0, y1, ns_start, ns_end):
        super(displacementGeneratorInterior, self).__init__(label, labelSample, ns_start, ns_end)
        
        self.npoints = Nx*Ny
        xv, yv = np.meshgrid( np.linspace(x0,x1,Nx) , np.linspace(y0,y1,Ny)  )
        self.x_eval = np.concatenate( (xv.reshape(self.npoints,1) , yv.reshape(self.npoints,1)), axis = 1 ) 

        
        