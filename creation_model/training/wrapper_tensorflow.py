from functools import partial, update_wrapper
import tensorflow as tf
import numpy as np  
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint

# Activations
dictActivations = {'tanh' : tf.nn.tanh, 
                    'sigmoid' : tf.nn.sigmoid , 
                    'linear': tf.keras.activations.linear,
                    'relu': tf.keras.activations.relu,
                    'leaky_relu': tf.nn.leaky_relu,
                    'swish' : tf.keras.activations.swish,
                    'elu' : tf.keras.activations.elu}


# ====================  Initiatialisations ==================================
dfInitK = tf.keras.initializers.glorot_uniform(seed = 1)
# dfInitK = tf.keras.initializers.VarianceScaling(scale=1.0, 
# mode="fan_in", distribution="untruncated_normal", seed=None)
dfInitB = tf.keras.initializers.Zeros()


# Display training progress by printing a single dot for each completed epoch
class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
         if epoch % 100 == 0: print('')
         print('.', end='')


# Custom Checkpoint
def checkpoint(saveFile, stepEpochs = 1, monitor = "val_loss"): 
    return ModelCheckpoint(saveFile, monitor='val_custom_loss_mse', verbose=1,
                           save_best_only=True, save_weights_only = True, mode='auto', period=stepEpochs)

# Custom scheduler for learning rate decay
def scheduler(epoch, decay, lr, EPOCHS):    
    omega = np.sqrt(float(epoch/EPOCHS))
    rate = lr*(1.0 - omega) + omega*decay*lr
    print('learning_rate = ', rate)
    return rate

def scheduler_linear(epoch, decay, lr, EPOCHS):    
    omega = float(epoch/EPOCHS)
    rate = lr*(1.0 - omega) + omega*decay*lr
    print('learning_rate = ', rate)
    return rate

def scheduler_cosinus(epoch, decay, lr, EPOCHS):  
    lr_mean = lr
    lr_min = lr*decay
    lr_amp = lr_mean - lr_min
    
    
    omega = np.sin(30*float(epoch/EPOCHS))
    rate = lr_min + lr_amp*omega
    print('learning_rate = ', rate, omega)
    return rate


# Custom weighted mse     
def custom_loss_mse(y_p,y_d,weight):
    return tf.reduce_sum(tf.reduce_mean(tf.multiply(weight,tf.square(tf.subtract(y_p,y_d))), axis=0))




def set_seed_default_initialisers(seed = 1):
    return tf.keras.initializers.glorot_uniform(seed = seed), tf.keras.initializers.Zeros()
    # return tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_in", 
    # distribution="untruncated_normal", seed=seed), tf.keras.initializers.Zeros()

# ============================================================================


# Improved partial: For some purposes the function should be labeled
def my_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

# Plotting history of training
def plot_history(history,label=['loss','val_loss'], savefile = None):
    plt.figure(1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(history.epoch, np.array(history.history[label[0]]), label='Train Loss')
    plt.plot(history.epoch, np.array(history.history[label[1]]), label = 'Val loss')
    plt.yscale('log')
    plt.legend()
    #  plt.ylim([0, ])
      
    if savefile:
        plt.savefig(savefile + '.png')
        np.savetxt(savefile + '_history_.txt', np.array(history.history[label[0]]) )
        np.savetxt(savefile + '_history_val.txt', np.array(history.history[label[1]]) )
      
    plt.grid()    
      # plt.show( )


