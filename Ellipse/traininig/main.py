import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0,'../')
sys.path.insert(0,'../utils/')
import matplotlib.pyplot as plt
import numpy as np

import myTensorflow as mytf
from timeit import default_timer as timer

import h5py
import pickle
import Generator as gene


base_offline_folder = '.'
folderData = '../generationDataEllipseEx/simuls/'

f = h5py.File(folderData + 'dataset.hdf5', 'r')
X_train = np.array(f['Train/X_disp'])
Y_train = np.array(f['Train/Y_stress'])
f.close()

Y_train = np.concatenate((Y_train[:,0:3],Y_train[:,-3:]), axis=1) # Constrain just to two points

ger = gene.displacementGenerator(['Right','Bottom','Top'],[10,10,10], 0.05, 8000, 10000)

x_e = ger.x_eval

# X_train_new = np.zeros((10000,435))

# for i in range(10000):
#     disp = X_train[i,:].reshape((30,2))
    
#     y = x_e + disp
    
#     kk = 0
#     for j in range(30):
#         for k in range(j+1,30):
#             X_train_new[i,kk] = np.linalg.norm(y[j,:] - y[k,:])
#             kk += 1
            
# with open('X_train_new.dat', 'wb') as f:
#     pickle.dump(X_train_new, f)   

Nin = X_train.shape[1]
Nout = Y_train.shape[1]

mytf.tf.set_random_seed(3)

Run_id = ''
EPOCHS = 500

weight_mu = 0.0

num_parameters = 0 # parameters that are taking into account into the model for minimization.

start = timer()

# Run_id = '0' # DNN (reduced feature) Adadelta, lr = 1.0, decay = 1.e-1 , [64,64,64,64,64], [0.0,0.0,0.0,0.0,0.0], lr2 = 0.0, actLabel = 'linear'
# model = mytf.DNNmodel(Nin, Nout, 64, actLabel = 'linear' )


# Run_id = '1' # 64, lr = 0.001, decay = 1.e-1 # second in performance, but with acceptable validation Error
# Run_id = '2' # 64, lr = 0.01, decay = 1.e-1
# Run_id = '3' # 256, lr = 0.0001, decay = 1.e-1 # third in perfomance 
# Run_id = '4' # 256, lr = 0.001, decay = 1.e-1 # optimal but with bad validation
# Run_id = '5' # 256, lr = 0.01, decay = 1.e-1 # with distances as X_train
# Run_id = '6' # 64, lr = 0.001, decay = 1.e-1 # with distances as X_train

# Run_id = '7' # lr = 0.001, decay = 1.e-1 , 64, actLabel = 'linear'

# Run_id = '8' # DNN_gen , lr = 1.0, decay = 1.e-1 , [256,256,256,256,256], [0.2,0.2,0.2,0.2,0.2], lr2 = 0.0, actLabel = 'linear'

# Run_id = '9' # (now doing test with zeros parameters) DNN_gen , lr = 0.001, decay = 1.e-1 , [256,256,256,256,256], [0.5,0.5,0.5,0.5,0.5], lr2 = 0.00001, actLabel = 'linear'

# Run_id = '10' # DNN_gen , lr = 0.001, decay = 1.e-1 , [256,256,256,256,256], 5*[0.0], lr2 = 0.00001, actLabel = 'linear'

# Run_id = '11' # continuation 10

# Run_id = '12' # continuation 10 with dropout [0.2,0.2,0.2,0.2,0.2]

# Run_id = '13' # DNN_gen , lr = 0.1, decay = 1.e-1 , [1024,1024,1024,1024], [0.3,0.3,0.3,0.3], lr2 = 0.0, actLabel = 'linear'
# # model = mytf.DNNmodel(Nin, Nout, 64, actLabel = 'linear' )

# Run_id = '14' # DNN simple , lr = 0.001, decay = 1.e-1 , [12,12,12,12], [0.0,0.0,0.0,0.0], lr2 = 0.0, actLabel = 'linear'
# # model = mytf.DNNmodel(Nin, Nout, 64, actLabel = 'linear' )


# Run_id = '15' # DNN (reduced feature), lr = 0.001, decay = 1.e-1 , [64,64,64,64,64], [0.0,0.0,0.0,0.0,0.0], lr2 = 0.0, actLabel = 'linear'
# # model = mytf.DNNmodel(Nin, Nout, 64, actLabel = 'linear' )


Neurons= [64,64,64,64,64]
drps =[0.0,0.0,0.0,0.0,0.0]
lr2 = 0.0
model = mytf.DNNmodel(Nin, Nout, Neurons, actLabel = 'linear', drps = drps, lambReg = lr2  )

# model.load_weights('weights_10')
    
history = mytf.my_train_model( model, X_train, Y_train, num_parameters, EPOCHS, lr = 1.0, decay = 1.e-1, w_l = 1.0, w_mu = 0.01)
    
mytf.plot_history( history)

with open('history_' + Run_id + '.dat', 'wb') as f:
    pickle.dump(history.history, f)
    

# model.save('saves_model' + Run_id + '.hd5')

model.save_weights('weights_' + Run_id)


end = timer()
print(end - start) # Time in seconds, e.g. 5.38091952400282

