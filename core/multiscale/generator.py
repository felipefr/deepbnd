import numpy as np
import core.fenics.io_wrappers as iofe
import dolfin as df

class dataGenerator:
    def __init__(self, label, labelSample, ns_start, ns_end):
        self.label = label
        self.labelSample = labelSample
        self.ns_start = ns_start
        self.ns_end = ns_end
        self.ns = self.ns_end - self.ns_start
    
    def __call__(self, data, Mesh, pm, RequiredFields, defaultMeshParam):
        pass
    

    
class displacementGenerator(dataGenerator):
    def __init__(self, label, labelSample, ns_start, ns_end):
        super(displacementGenerator, self).__init__(label, labelSample, ns_start, ns_end)
        
        self.npoints = 0
        self.x_eval = np.zeros((self.npoints,2))
        
    
    def __call__(self, Mesh, u):       
        
        data = np.zeros((2*self.npoints, self.ns))
        
        for i in range(self.ns_start, self.ns_end):
            print("Generating data snapshot displacement {0}.{1}".format(self.labelSample,i))
            
            boundTree = Mesh.bounding_box_tree().build(Mesh) # force reconstruction
            
            for j in range(self.npoints):
                for k in range(2):
                    data[2*j + k, i-self.ns_start] = u[k](self.x_eval[j,:]) 
            
            del boundTree
        
        return data

    
    def identifyNodes(self,mesh): # run only if it matches the vertices
        tol = 1.0e-8
        # V = df.VectorFunctionSpace(mesh,"CG", 1) 
        # X = V.tabulate_dof_coordinates()[::2,:]        
        X = mesh.coordinates()
        return np.array([np.where( ( abs(X[:,0] - xi[0])<tol ) & ( abs(X[:,1] - xi[1])<tol ))[0][0] for xi in self.x_eval])

    
    def shape(self):
        return (self.ns,2*self.npoints)


class displacementGeneratorBoundary(displacementGenerator):
    def __init__(self, x0, y0, Lx, Ly, NpointsFace):
        super(displacementGeneratorBoundary, self).__init__(label = 'default' , labelSample = 'default', ns_start = 0, ns_end=1)
        
        self.npoints = 4*NpointsFace - 4
        self.x_eval = np.zeros((self.npoints,2))
    
        N = NpointsFace - 1
        hx = Lx/N
        hy = Ly/N
        
        self.x_eval[:N,1] = y0
        self.x_eval[:N,0] = np.linspace(x0, x0 + Lx - hx, N)
        
        self.x_eval[N:2*N,0] = x0 + Lx
        self.x_eval[N:2*N,1] = np.linspace(y0, y0 + Ly - hy, N)
    
        self.x_eval[2*N:3*N,1] = y0 + Ly 
        self.x_eval[2*N:3*N,0] = np.linspace(x0 + Lx, x0 + hx ,N )
        
        self.x_eval[3*N:,0] = x0
        self.x_eval[3*N:,1] = np.linspace(y0 + Ly, y0 + hy, N)
        
        
        