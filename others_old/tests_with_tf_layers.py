
import tensorflow as tf
from tensorflow import keras
import numpy as np

print(tf.__version__)

tf.enable_eager_execution()

tfe = tf.contrib.eager

class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(MyDenseLayer, self).__init__()
    self.num_outputs = num_outputs
    
  def build(self, input_shape):
    self.kernel = self.add_variable("kernel", 
                                    shape=[input_shape[-1].value, 
                                           self.num_outputs])
    
  def call(self, input):
    return tf.matmul(input, self.kernel)
  
layer = MyDenseLayer(10)
print(layer(tf.zeros([10, 5])))
print(layer.variables)



layer2 = tf.keras.layers.Dense(10, input_shape=(None, 5))
layer2(tf.zeros([10, 5]))
layer2( tf.eye( 5 ) )
print(layer2.variables)
layer2.kernel = tf.zeros( layer2.kernel.shape )
print( layer2.bias )
layer2.bias  = layer2.bias + 1 



#%%


import tensorflow as tf
import numpy as np
tf.enable_eager_execution()









