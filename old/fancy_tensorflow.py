# Composition of sequence of layers
def mySequential(y,layers):
    
    for l in layers:
        y = l(y)
    
    return y

# Composition of a sequence of alternated layers (layer2_0(layer1_0) .. )
def mySequential_twoLayers(y,layers1,layers2):
    
    for l1,l2 in zip(layers1,layers2):
        y = l1(y)
        y = l2(y)
    
    return y


class MLPblock(tf.keras.layers.Layer):
    def __init__(self, Nneurons, input_shape): 
        super(MLPblock, self).__init__()

      
        self.hidden_layers = [
        tf.keras.layers.Dense( Nneurons, activation=tf.keras.activations.linear,
                            input_shape=input_shape),
        tf.keras.layers.Dense( Nneurons, activation=tf.nn.relu),
        tf.keras.layers.Dense(Nneurons, activation=tf.nn.leaky_relu),
        tf.keras.layers.Dense(Nneurons, activation=tf.nn.relu),
        tf.keras.layers.Dense(Nneurons, activation=tf.nn.relu)]
        
    def call(self, input_features):
        return mySequential(input_features,self.hidden_layers)


class MLPblock_gen(tf.keras.layers.Layer):
    def __init__(self, Nneurons, input_shape, drps, lambReg, actLabel): 
        super(MLPblock_gen, self).__init__()
        
        l2_reg = lambda w: lambReg * tf.linalg.norm(w)**2.0
        l1_reg = lambda w: lambReg * tf.linalg.norm(w,ord=1)
        
        reg = tf.keras.regularizers.l1_l2(l1=0.1*lambReg, l2=lambReg)
        
        self.nLayers = len(Nneurons)
        
        assert self.nLayers == len(drps) - 2 , "add dropout begin and end"
        
        self.ld = [tf.keras.layers.Dropout(drps[0])]
        self.la = [tf.keras.layers.Dense(Nneurons[0], activation=dictActivations[actLabel[0]], input_shape=input_shape, kernel_initializer=dfInitK, bias_initializer=dfInitB)]
            
        for n, drp in zip(Nneurons[1:],drps[1:-1]):
            self.ld.append(tf.keras.layers.Dropout(drp)) 
            self.la.append(tf.keras.layers.Dense(n, activation=dictActivations[actLabel[1]], 
                          kernel_regularizer=reg, bias_regularizer=reg, 
                          kernel_constraint=tf.keras.constraints.MaxNorm(300.0), kernel_initializer=dfInitK, bias_initializer=dfInitB))   
        
        self.ld.append(tf.keras.layers.Dropout(drps[-1])) 

    def call(self, input_features):
        return mySequential_twoLayers(input_features,self.la, self.ld)
    

class DNNmodel(tf.keras.Model):
    def __init__(self, Nin, Nout, Neurons, actLabel, drps=None, lambReg=0.0):
        super(DNNmodel,self).__init__()
        
        self.Nin = Nin
        self.Nout = Nout
        
        if(type(drps) == type(None)):
            self.mlpblock = MLPblock(Nneurons=Neurons, input_shape=(self.Nin,))
        else:   
            self.mlpblock = MLPblock_gen(Neurons, (self.Nin,), drps, lambReg, actLabel[0:2])
            
        self.outputLayer = tf.keras.layers.Dense( Nout, activation=dictActivations[actLabel[2]], kernel_initializer=dfInitK, bias_initializer=dfInitB)
        
    def call(self, inputs):
        
        z = self.mlpblock(inputs)
        y = self.outputLayer(z)
        
        return y