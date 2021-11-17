import numpy as np
# import myLibRB as myrb

from sklearn.decomposition import TruncatedSVD, PCA
import matplotlib.pyplot as plt

norm1d = lambda x, m: (x - m[0])/m[1]
unnorm1d = lambda x, m: m[1]*x + m[0]
scale1d = lambda x, m: (x - m[0])/(m[1] - m[0]) 
unscale1d = lambda x, m: m[0] + x*(m[1] - m[0]) 
scaler = lambda foo, args: lambda X: np.array( [ [foo(Xij,mj) for (Xij, mj) in zip(Xi,args) ] for Xi in X ] ) 




class myTransfomation:
    def __init__(self, Xref, kind = 'minmax'):
        self.Xref = Xref
        self.subsets = {}
        self.kind = kind
        
        options = {'minmax': lambda x,s: scale1d(x,[s['max'],s['min']]),
                   'norm' : lambda x,s: norm1d(x,[s['mean'],s['std']]) }
        
        optionsInv = {'minmax': lambda x,s: unscale1d(x,[s['max'],s['min']]),
                      'norm' : lambda x,s: unnorm1d(x,[s['mean'],s['std']]) }
        
        self.transform1d = options[kind]
        self.invtransform1d = optionsInv[kind]
        
    def registerSubset(self, name, indexes=[]):
        if(len(indexes) == 0 ):
            indexes = np.arange(self.Xref.shape[1])
            
        self.subsets[name]={'indexes' : indexes}
        self.subsets[name]['max'] = np.max(self.Xref[:,indexes])
        self.subsets[name]['min'] = np.min(self.Xref[:,indexes])
        
        self.subsets[name]['mean'] = np.mean(self.Xref[:,indexes])
        self.subsets[name]['std'] = np.std(self.Xref[:,indexes])
        
    def showStats(self, X = None):
        
        if(type(X)!=type(None)):
            for k, s in self.subsets.items():
                ind = s['indexes']
                Xind = X[:,ind]
                print('subset=',k)
                print('max=',np.max(Xind))
                print('min=',np.min(Xind))
                print('mean=',np.mean(Xind))
                print('std=',np.std(Xind))
        else:
            for k, s in self.subsets.items():
                print('subset=',k)
                print('max=',s['max'])
                print('min=',s['min'])
                print('mean=',s['mean'])
                print('std=',s['std'])
        
    def transform(self, X):
        X_t = np.array(X)
        
        for s in self.subsets.values():
            X_t[:,s['indexes']] = self.transform1d(X[:,s['indexes']],s)  
            
        return X_t
    
    def inverse_transform(self, X):
        X_t = np.zeros_like(X) 
        for s in self.subsets.values():
            X_t[:,s['indexes']] = self.invtransform1d(X[:,s['indexes']],s)  
            
        return X_t
    
    
def getPCAcut(X, subsets, xr0min, xr0max, xr1min, xr1max):
    
    T = myTransfomation(X, 'minmax')
    for name, index in subsets.items():
        T.registerSubset(name, indexes = index)
    X_t = T.transform(X)
    
    svd = PCA(n_components=2, svd_solver = 'arpack')   
    svd.fit(X_t)
    X_r = svd.transform(X_t)
    
    
    ind1 = set(np.where(X_r[:,0] > xr0min)[0])
    ind2 = set(np.where(X_r[:,0] < xr0max)[0])
    ind3 = set(np.where(X_r[:,1] > xr1min)[0])
    ind4 = set(np.where(X_r[:,1] < xr1max)[0])
    ind = np.array(list((ind1 & ind2) & (ind3 & ind4)))
   
    plt.scatter(X_r[:,0],X_r[:,1])
    plt.grid()
    
    plt.show()
    

    return ind


def getCombinedPCAnormalisation(X, subsets, N):
    T1 = myTransfomation(X,'norm')
    for name, index in subsets.items():
        T1.registerSubset(name, indexes = index)
    X_t = T1.transform(X)

    svd = PCA(n_components=N, svd_solver = 'arpack')

    svd.fit(X_t)
    X_r = svd.transform(X_t)

    T2 = myTransfomation(X_r, 'minmax')  
    [ T2.registerSubset(str(i), indexes = [i] ) for i in range(N)]
    
    return T2.transform(X_r) , [T1,svd,T2]

def getCombinedPCAreconstruction(X,T):
        X_1 = T[2].inverse_transform(X)
        X_0 = T[1].inverse_transform(X_1)
        
        return T[0].inverse_transform(X_0)



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
    
    
# class OfflineData:
    
#     def __init__(self, folder, nAff_A, nAff_f, isTest = False):
#         self.folder = folder
        
#         self.nodes = np.loadtxt(folder + "nodes.txt")
        
#         self.Nnodes, self.nDim = self.nodes.shape

#         if(isTest):
#             self.snapshots = np.loadtxt(folder + 'snapshotsTest.txt').transpose() # adequate for ML
#         else:
#             self.snapshots = np.concatenate( (np.loadtxt(folder + 'snapshotsFEA.txt') , np.loadtxt(folder + 'snapshotsRB.txt') ), axis = 1).transpose()

        
#         self.ns , self.Nh = self.snapshots.shape
        
#         self.nDof = int(self.Nh/self.Nnodes) 
        
#         self.rb = myrb.getRBmanager(nAff_A, nAff_f, folder)

#         if(isTest):
#             self.param = np.loadtxt(folder + 'paramTest.txt')
#         else:
#             self.param = np.concatenate( (np.loadtxt(folder + 'paramFEA.txt'), np.loadtxt(folder + 'paramRB.txt')), axis = 0)

#         self.param_min = np.min(self.param,axis = 0)
#         self.param_max = np.max(self.param,axis = 0)
        
#         self.num_parameters = self.param.shape[1]


# class DataGenerator:

#     def __init__(self, rawData):
#         self.rawData = rawData
#         self.data = {}
#         self.nodes_loc = {}
#         self.dofs_loc = {}
        
#     def buildFeatures(self, listFeatures = [], region = None,  label = 'in', Nsample = -1 , seed = 0): # Nnodes_loc <0 means that 
        
#         np.random.seed(seed)
        
#         N = 0 
        
#         admNodes = []
        
#         if(region != None):
#             admNodes = region.getAdmissibleNodes(self.rawData.nodes)
            
#         if(Nsample == -1): # selecting all nodes
#             Nsample = len(admNodes) 
#             self.nodes_loc[label] = admNodes
#         else:
#             Nsample = min(len(admNodes) , Nsample)
#             self.nodes_loc[label] = np.sort(np.random.choice(admNodes,Nsample, replace = False))

#         if(Nsample>0):
#             self.dofs_loc[label] = np.zeros(self.rawData.nDim*Nsample, dtype = 'int')
            
#             for i in range(self.rawData.nDim):
#                 self.dofs_loc[label][i::self.rawData.nDim] = self.rawData.nDim*self.nodes_loc[label] + i 
        
#         for featureType in listFeatures:
            
#             if(featureType == 'u' or featureType == 'x' or featureType == 'x+u'):
#                 N += self.rawData.nDim*max(Nsample,0)
#             elif(featureType == 'mu'):
#                 N += self.rawData.num_parameters
#             else:
#                 print('feature type not found ', featureType[0])
#                 exit()
                

#         self.data[label] = np.zeros((self.rawData.ns,N))
        
#         k = 0
        
#         for featureType in listFeatures:
            
#             if(featureType == 'u'):
#                 l = self.rawData.nDim*max(Nsample,0)
#                 self.data[label][:,k:k+l] = self.rawData.snapshots[:,self.dofs_loc[label]]
#                 k = k + l 
                
#             elif(featureType == 'x'):
#                 l = self.rawData.nDim*max(Nsample,0)
#                 nodesTemp = self.rawData.nodes[self.nodes_loc[label],:].flatten()
#                 self.data[label][:,k:k+l] = nodesTemp
#                 k = k + l 
                
#             elif(featureType == 'x+u'):
#                 l = self.rawData.nDim*max(Nsample,0)
#                 nodesTemp = self.rawData.nodes[self.nodes_loc[label],:].flatten()
#                 self.data[label][:,k:k+l] = self.rawData.snapshots[:,self.dofs_loc[label]] + nodesTemp
#                 k = k + l 
                
#             elif(featureType == 'mu'):
#                 l = self.rawData.num_parameters
#                 self.data[label][:,k:k+l] = self.rawData.param
#                 k = k + l
#             else:
#                 print('feature type not found ', featureType[0])
#                 exit()
                
#     def normaliseFeatures(self,label, indexes, minV, maxV):
        
#         for i,j in enumerate(indexes):
#             self.data[label][:, j] = (self.data[label][:, j]  - minV[i])/(maxV[i] - minV[i]) 
            
            
            