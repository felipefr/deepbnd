#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 07:30:57 2020

@author: felipefr
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

histories_tf = []
histories_pde = []
histories_tf_wmu0 = []

epochs = 500

# histories_tf_mean = {'loss':np.zeros(epochs), 'val_loss':np.zeros(epochs), 'mae_mu': np.zeros(epochs),'val_mae_mu':np.zeros(epochs), 'mae_loc': np.zeros(epochs),'val_mae_loc':np.zeros(epochs) }
# histories_pde_mean = {'loss':np.zeros(epochs), 'val_loss':np.zeros(epochs), 'mae_mu': np.zeros(epochs),'val_mae_mu':np.zeros(epochs), 'mae_loc': np.zeros(epochs),'val_mae_loc':np.zeros(epochs) }

folder = 'saves_3param/'



for i in range(1,11):
    histories_tf.append(pickle.load(open(folder + 'historyModel_64_' + str(i) + '.dat','rb')))
    histories_pde.append(pickle.load(open(folder + 'historyModel_64_pde_' + str(i) + '.dat','rb')))
    
    
    # histories_tf[-1]['loss'] = histories_tf[-1]['loss']/histories_tf[-1]['loss'][2]
    # histories_tf[-1]['val_loss'] = histories_tf[-1]['val_loss']/histories_tf[-1]['val_loss'][2]

    # histories_pde[-1]['loss'] = histories_pde[-1]['loss']/histories_pde[-1]['loss'][2]
    # histories_pde[-1]['val_loss'] = histories_pde[-1]['val_loss']/histories_pde[-1]['val_loss'][2]
        
        # histories_tf_mean[s] +=  histories_tf[-1][s]
        
# for s in histories_tf_mean.keys():
#     histories_tf_mean[s] =  histories_tf_mean[s]

folder = 'saves_test/'
for i in range(1,2):
    histories_tf_wmu0.append(pickle.load(open(folder + 'historyModel_' + str(i) + '.dat','rb')))

    
plt.figure(1,(10,5))

# for one tf mae_mu
plt.subplot('221')
for i in range(1,2):
    plt.plot(histories_tf[i]['mae_mu'])
    plt.plot(histories_tf[i]['val_mae_mu'])

plt.plot(histories_tf_wmu0[0]['mae_mu'])
    
plt.yscale('log')
plt.xlabel('epochs')
plt.ylabel('mae_mu tf')
plt.grid()
# plt.yticks([1.e-2,1.e-3,1.e-4,1.e-5])
# plt.yticks([1.e-2,1.e-3,1.e-4,1.e-5,1.e-6,1.e-7,1.e-8])

plt.subplot('222')

# for one pde mae_mu
for i in range(1,2):
    plt.plot(histories_pde[i]['mae_mu'])
    plt.plot(histories_pde[i]['val_mae_mu'])

plt.yscale('log')
plt.xlabel('epochs')
plt.ylabel('mae_mu pde')
# plt.yticks([1.e-1,1.e-2,1.e-3])
# plt.yticks([1.e-2,1.e-3,1.e-4,1.e-5,1.e-6,1.e-7,1.e-8])
plt.grid()


# for all tf mae_mu
plt.subplot('223')
for i in range(1,9):
    plt.plot(histories_tf[i]['mae_mu'])
    plt.plot(histories_tf[i]['val_mae_mu'])
    
plt.yscale('log')
plt.xlabel('epochs')
plt.ylabel('mae_mu tf')
plt.grid()
# plt.yticks([1.e-2,1.e-3,1.e-4,1.e-5])
# plt.yticks([1.e-2,1.e-3,1.e-4,1.e-5,1.e-6,1.e-7,1.e-8])

plt.subplot('224')

# for all pde mae_mu
for i in range(1,9):
    plt.plot(histories_pde[i]['mae_mu'])
    plt.plot(histories_pde[i]['val_mae_mu'])

plt.yscale('log')
plt.xlabel('epochs')
plt.ylabel('mae_mu pde')
# plt.yticks([1.e-1,1.e-2,1.e-3])
# plt.yticks([1.e-2,1.e-3,1.e-4,1.e-5,1.e-6,1.e-7,1.e-8])
plt.grid()

plt.tight_layout()

# plt.savefig(folder + "mae_loc_tf_pde.png")
plt.show()