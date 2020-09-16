import numpy as np
import h5py

class genericParams:
    def __init__(self, name = '', fileName = None):
        self.variables = {}
        self.fileName = fileName
        self.name = name
        
    def addVariable(self,label,v):
        self.variables[label] = v
    
    def __call__(self,varLabels=[]):
        if(varLabels == []):
            return np.array(self.variables.values())
        elif(type(varLabels) ==type(None)):
            return []
        else:
            return np.array([self.variables[l] for l in varLabels])
        
class randomParams(genericParams):
    
    def __init__(self,seed=1, name = '', fileName=None):
        super(randomParams,self).__init__(name,fileName)
        self.seed = seed
        np.random.seed(self.seed)
        self.samples = {}
        self.seeds = {}
        self.randomCounter = 0
        
    def addVariable(self,label,vmin,vmax):
        self.variables[label] = {'min':vmin, 'max': vmax}

    def addSample(self,label,ns, seed = None, shift = 0 ):
        self.seeds[label] = (seed if type(seed) != type(None) else self.seed, self.randomCounter + shift)
        
        self.setSeed(seed, shift)
       
        sample = np.array([[ v['min'] + np.random.uniform()*(v['max'] - v['min']) for v in self.variables.values()] for i in range(ns)])
        self.samples[label] = sample        
        self.randomCounter += ns*len(self.variables) +  shift
        
        for i,v in enumerate(self.variables.values()): # just create a view (don't copy data)
            v[label] = sample[:,i]
            
    def setSeed(self,seed, shift):
        if(type(seed) != type(None)):
            self.seed = seed
            np.random.seed(self.seed)
            self.randomCounter = shift
        
        for i in range(shift):
            np.random.uniform()
        
    def __call__(self,i,sampleLabel,varLabels=[]):
        if(varLabels == []):
            return self.samples[sampleLabel][i,:]
        elif(type(varLabels) == type(None)):
            return []
        else:
            return np.array([self.variables[l][sampleLabel][i] for l in varLabels])
        
    def write(self, mode = 'a'):
        with  h5py.File(self.fileName, mode) as f:
            for ls,s in self.samples.items():
                g = f.create_group(ls)
                g.attrs['seed'] = self.seeds[ls][0]
                g.attrs['shift'] = self.seeds[ls][1]
                g.create_dataset('sample',data=s, compression = "gzip")
      

class derivedParams(genericParams):
        
    def __init__(self, pr, pc, name = '', fileName = None):
        super(derivedParams,self).__init__(name, fileName)
        self.pr = pr
        self.pc = pc 
 
    def addVariable(self,label,listVar_pr, listVar_pc, foo):
        self.variables[label] = {'listVar_pr' : listVar_pr, 'listVar_pc' : listVar_pc, 'foo' : foo}

    def __call__(self,i,sampleLabel):
        listVar_pr = lambda v: list(self.pr(i,sampleLabel,v['listVar_pr']))
        listVar_pc = lambda v: list(self.pc(v['listVar_pc']))
       
        return np.array([v['foo'](*(listVar_pr(v) + listVar_pc(v))) for v in self.variables.values()])