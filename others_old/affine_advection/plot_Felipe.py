import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

histories_tf = []
histories_pde = []

epochs = 500

histories_tf_mean = {'loss':np.zeros(epochs), 'val_loss':np.zeros(epochs), 'mean_absolute_error': np.zeros(epochs),'val_mean_absolute_error':np.zeros(epochs) }
histories_pde_mean = {'loss':np.zeros(epochs), 'val_loss':np.zeros(epochs), 'mean_absolute_error': np.zeros(epochs),'val_mean_absolute_error':np.zeros(epochs) }

for i in range(1,11):
    histories_tf.append(pickle.load(open('./saves_lr0001/historyModel_' + str(i) + '.dat','rb')))
    for s in histories_tf_mean.keys():
        histories_tf_mean[s] +=  histories_tf[-1][s]
        
for s in histories_tf_mean.keys():
    histories_tf_mean[s] =  histories_tf_mean[s]/float(2)

for i in range(1,11):
    histories_pde.append(pickle.load(open('./saves_lr0001/historyModel_pde_' + str(i) + '.dat','rb')))
    for s in histories_pde_mean.keys():
        histories_pde_mean[s] +=  histories_pde[-1][s]
        
for s in histories_pde_mean.keys():
    histories_pde_mean[s] =  histories_pde_mean[s]/float(1)
    
    
plt.figure(1,(10,5))

# all for tf mean_absolute_error
plt.subplot('221')
for i in range(10):
    plt.plot(histories_tf[i]['mean_absolute_error'])
    plt.plot(histories_tf[i]['val_mean_absolute_error'])
    
plt.yscale('log')
plt.xlabel('epochs')
plt.ylabel('mean_absolute_error tf')
plt.grid()
# plt.yticks([1.e-2,1.e-3,1.e-4,1.e-5])
# plt.yticks([1.e-2,1.e-3,1.e-4,1.e-5,1.e-6,1.e-7,1.e-8])

plt.subplot('222')

# mean for tf mae
plt.plot(histories_tf_mean['mean_absolute_error'])
plt.plot(histories_tf_mean['val_mean_absolute_error'])

plt.yscale('log')
plt.xlabel('epochs')
plt.ylabel('mean_absolute_error tf')
# plt.yticks([1.e-1,1.e-2,1.e-3])
# plt.yticks([1.e-2,1.e-3,1.e-4,1.e-5,1.e-6,1.e-7,1.e-8])
plt.grid()



# all for pde mean_absolute_error
plt.subplot('223')
for i in range(10):
    plt.plot(histories_pde[i]['mean_absolute_error'])
    plt.plot(histories_pde[i]['val_mean_absolute_error'])
    
plt.yscale('log')
plt.xlabel('epochs')
plt.ylabel('mean_absolute_error pde')
plt.grid()
# plt.yticks([1.e-2,1.e-3,1.e-4,1.e-5,1.e-6,1.e-7,1.e-8])

plt.subplot('224')

# mean for pde  mae
plt.plot(histories_pde_mean['mean_absolute_error'])
plt.plot(histories_pde_mean['val_mean_absolute_error'])

plt.yscale('log')
plt.xlabel('epochs')
plt.ylabel('mean_absolute_error pde')
# plt.yticks([1.e-1,1.e-2,1.e-3,1.e-4])
# plt.yticks([1.e-2,1.e-3,1.e-4,1.e-5,1.e-6,1.e-7,1.e-8])
plt.grid()


plt.savefig("./figs/lr0001_mae.png")
plt.show()