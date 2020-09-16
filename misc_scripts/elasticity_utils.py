import numpy as np
import myLibRB as myrb

lame2youngPoisson  = lambda lamb, mu : [ 0.5*lamb/(mu + lamb) , mu*(3.*lamb + 2.*mu)/(lamb + mu) ]
youngPoisson2lame = lambda nu,E : [ nu * E/((1. - 2.*nu)*(1.+nu)) , E/(2.*(1. + nu)) ]



def youngPoisson2lame_planeStress(nu,E):
    lamb , mu = youngPoisson2lame(nu,E)
    
    lamb = (2.0*mu*lamb)/(lamb + 2.0*mu)
    
    return lamb, mu

lame2lameStar = lambda lamb, mu: [(2.0*mu*lamb)/(lamb + 2.0*mu), mu]
lameStar2lame = lambda lambStar, mu: [(2.0*mu*lambStar)/(-lambStar + 2.0*mu), mu]

composition = lambda f,g: lambda x,y : f(*g(x,y))

convertParam2 = lambda p,f: np.array( [  f(*p_i) for p_i in p ] )

def convertParam(param,foo):
    
    n = len(param)
    paramNew = np.zeros((n,2))
    for i in range(n):
        paramNew[i,0], paramNew[i,1] = foo( *param[i,:].tolist()) 
  
    return paramNew

scale1d = lambda x, m: (x - m[0])/(m[1] - m[0]) 
unscale1d = lambda x, m: m[0] + x*(m[1] - m[0]) 
scaler = lambda foo, args: lambda X: np.array( [ [foo(Xij,mj) for (Xij, mj) in zip(Xi,args) ] for Xi in X ] )       


class Box:

    def __init__(self,x0,x1,y0,y1):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1

    def getAdmissibleNodes(self,nodes):
        return np.where(np.logical_and(np.logical_and(nodes[:,0]>self.x0,nodes[:,0]<self.x1), np.logical_and(nodes[:,1]>self.y0,nodes[:,1]<self.y1)) )[0]
       
    
class Region:
    def __init__(self, boxes):
        self.boxes = boxes
            
    def getAdmissibleNodes(self,nodes):
        
        indexesList = [] 
        
        for b in self.boxes:
            indexesList+=list(b.getAdmissibleNodes(nodes))

        return np.array(indexesList)
    
    
class OfflineData:
    
    def __init__(self, folder, nAff_A, nAff_f, isTest = False):
        self.folder = folder
        
        self.nodes = np.loadtxt(folder + "nodes.txt")
        
        self.Nnodes, self.nDim = self.nodes.shape

        if(isTest):
            self.snapshots = np.loadtxt(folder + 'snapshotsTest.txt').transpose() # adequate for ML
        else:
            self.snapshots = np.concatenate( (np.loadtxt(folder + 'snapshotsFEA.txt') , np.loadtxt(folder + 'snapshotsRB.txt') ), axis = 1).transpose()

        
        self.ns , self.Nh = self.snapshots.shape
        
        self.nDof = int(self.Nh/self.Nnodes) 
        
        self.rb = myrb.getRBmanager(nAff_A, nAff_f, folder)

        if(isTest):
            self.param = np.loadtxt(folder + 'paramTest.txt')
        else:
            self.param = np.concatenate( (np.loadtxt(folder + 'paramFEA.txt'), np.loadtxt(folder + 'paramRB.txt')), axis = 0)

        self.param_min = np.min(self.param,axis = 0)
        self.param_max = np.max(self.param,axis = 0)
        
        self.num_parameters = self.param.shape[1]


class DataGenerator:

    def __init__(self, rawData):
        self.rawData = rawData
        self.data = {}
        self.nodes_loc = {}
        self.dofs_loc = {}
        
    def buildFeatures(self, listFeatures = [], region = None,  label = 'in', Nsample = -1 , seed = 0): # Nnodes_loc <0 means that 
        
        np.random.seed(seed)
        
        N = 0 
        
        admNodes = []
        
        if(region != None):
            admNodes = region.getAdmissibleNodes(self.rawData.nodes)
            
        if(Nsample == -1): # selecting all nodes
            Nsample = len(admNodes) 
            self.nodes_loc[label] = admNodes
        else:
            Nsample = min(len(admNodes) , Nsample)
            self.nodes_loc[label] = np.sort(np.random.choice(admNodes,Nsample, replace = False))

        if(Nsample>0):
            self.dofs_loc[label] = np.zeros(self.rawData.nDim*Nsample, dtype = 'int')
            
            for i in range(self.rawData.nDim):
                self.dofs_loc[label][i::self.rawData.nDim] = self.rawData.nDim*self.nodes_loc[label] + i 
        
        for featureType in listFeatures:
            
            if(featureType == 'u' or featureType == 'x' or featureType == 'x+u'):
                N += self.rawData.nDim*max(Nsample,0)
            elif(featureType == 'mu'):
                N += self.rawData.num_parameters
            else:
                print('feature type not found ', featureType[0])
                exit()
                

        self.data[label] = np.zeros((self.rawData.ns,N))
        
        k = 0
        
        for featureType in listFeatures:
            
            if(featureType == 'u'):
                l = self.rawData.nDim*max(Nsample,0)
                self.data[label][:,k:k+l] = self.rawData.snapshots[:,self.dofs_loc[label]]
                k = k + l 
                
            elif(featureType == 'x'):
                l = self.rawData.nDim*max(Nsample,0)
                nodesTemp = self.rawData.nodes[self.nodes_loc[label],:].flatten()
                self.data[label][:,k:k+l] = nodesTemp
                k = k + l 
                
            elif(featureType == 'x+u'):
                l = self.rawData.nDim*max(Nsample,0)
                nodesTemp = self.rawData.nodes[self.nodes_loc[label],:].flatten()
                self.data[label][:,k:k+l] = self.rawData.snapshots[:,self.dofs_loc[label]] + nodesTemp
                k = k + l 
                
            elif(featureType == 'mu'):
                l = self.rawData.num_parameters
                self.data[label][:,k:k+l] = self.rawData.param
                k = k + l
            else:
                print('feature type not found ', featureType[0])
                exit()
                
    def normaliseFeatures(self,label, indexes, minV, maxV):
        
        for i,j in enumerate(indexes):
            self.data[label][:, j] = (self.data[label][:, j]  - minV[i])/(maxV[i] - minV[i]) 
                
   
        