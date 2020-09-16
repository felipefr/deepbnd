#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 12:04:17 2020

@author: felipefr
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

u_ex = lambda alpha : 2.0/alpha

np.random.seed(3)

ns = 500
noise = 0.001
number_of_inputs = 1
number_of_output = 1
lr = 0.01
EPOCHS = 500
Nneurons = 32

alpha_min = 10.0
alpha_max = 11.0

x_train = alpha_min + np.random.rand(ns)*(alpha_max-alpha_min)
y_train = u_ex(x_train) + noise*(np.random.rand(ns)-0.5)

x_train = x_train.reshape((ns,1))
y_train = y_train.reshape((ns,1))

# scalerX = MinMaxScaler()
# scalerY = MinMaxScaler()

# scalerX.fit(x_train)
# scalerY.fit(y_train)

# x_train_norm = scalerX.transform(x_train)
# y_train_norm = scalerY.transform(y_train)

tf.set_random_seed(1234)
model = keras.Sequential([
keras.layers.Dense( Nneurons, activation=tf.keras.activations.linear,
                    input_shape=(number_of_inputs,)),
keras.layers.Dense( Nneurons, activation=tf.nn.relu),
keras.layers.Dense(Nneurons, activation=tf.nn.leaky_relu),
keras.layers.Dense(Nneurons, activation=tf.nn.relu),
keras.layers.Dense(Nneurons, activation=tf.nn.relu),
keras.layers.Dense( number_of_output, activation=tf.keras.activations.linear)])

  
optimizer= tf.train.AdamOptimizer(learning_rate = lr, beta1=0.8, beta2=0.7, epsilon=0.01, use_locking=False)
# optimizer= tf.train.GradientDescentOptimizer(lr)

model.compile(loss='mse',optimizer=optimizer,metrics=['mae'])
    

history = model.fit(x_train, y_train, epochs=EPOCHS,validation_split=0.2, verbose=1,
                    callbacks=[PrintDot()], batch_size = 32)



plt.figure()
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.plot(history.epoch, np.array(history.history['loss']),label='Train Loss')
plt.plot(history.epoch, np.array(history.history['val_loss']),label = 'Val loss')
plt.yscale('log')


