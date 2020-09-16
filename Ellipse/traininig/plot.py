import matplotlib.pyplot as plt
import numpy as np
import pickle

p = [pickle.load(open('history_{0}.dat'.format(i),'rb')) for i in range(1,16)]
# p += [pickle.load(open('history_{0}.dat'.format(i),'rb')) for i in range(14,16)]

p[10]['loss'] = np.concatenate((p[9]['loss'],p[10]['loss']))  
p[11]['loss'] = np.concatenate((p[9]['loss'],p[11]['loss']))

p[10]['val_loss'] = np.concatenate((p[9]['val_loss'],p[10]['val_loss']))  
p[11]['val_loss'] = np.concatenate((p[9]['val_loss'],p[11]['val_loss']))
  
plt.figure(1)

for i,pi in enumerate([p[12]]):
    plt.plot(pi['loss'], label = str(i))
    plt.plot(pi['val_loss'], label = 'val' + str(i))

plt.legend()
plt.grid()
plt.yscale('log')
plt.ylabel('mse')
plt.ylim(0.01,10.0)
plt.show()
