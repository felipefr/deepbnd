from micro_macro_data import read_micro_macro_data

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
tf.keras.backend.set_floatx('float32') # 32 bits seems to be enough for now.
from datetime import datetime

def create_nn(n_features, n_labels, n_layers, n_neurons_internal, learning_rate=1.0e-3, end_sigmoid=True):

  kernel_initializer = keras.initializers.he_normal(seed=None)
  bias_initializer = keras.initializers.Zeros()
  activation_internal = activation=tf.nn.relu

  if end_sigmoid:
    activation_end = activation=tf.nn.sigmoid
  else:
    activation_end = None

  model = keras.Sequential()

  # Initial layer.
  model.add(keras.layers.Dense(n_neurons_internal, activation=activation, input_shape=[n_features], kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))

  # Internal layers.
  for i in range(n_layers - 1):
    model.add(keras.layers.Dense(n_neurons_internal, activation=activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))

  # Last layers (without a relu -> output always positive).
  model.add(keras.layers.Dense(n_labels, activation=activation_end, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))

  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  # optimizer = 'adam'
  # optimizer = 'sgd'
  # optimizer = 'adadelta'

  # Generation of the network.
  model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error'])

  return model

def train_nn(n_rb_modes, training_ratio, n_layers, n_neurons_internal, validation_split, number_of_epochs, learning_rate, normalize_output=True):

  training_inputs, training_labels = read_micro_macro_data(n_rb_modes=n_rb_modes, normalize_output=normalize_output, ratio_samples=training_ratio)
  n_features = training_inputs.shape[1]
  n_labels = training_labels.shape[1]

  nn = create_nn(n_features, n_labels, n_layers, n_neurons_internal, learning_rate, end_sigmoid=normalize_output)
  nn.summary()

  run_id = 0
  time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
  model_output = "trained_models/nn_n_rb_modes_%d_n_internal_layers_%d_n_neurons_%d_adam_learning_rate_%.1e_n_epochs_%d_run_id_%d_%f_cases_%s.h5" % (n_rb_modes, n_layers, n_neurons_internal, learning_rate, number_of_epochs, run_id, training_ratio * 100, time_str)
  history_output = "logs/nn_n_rb_modes_%d_n_internal_layers_%d_n_neurons_%d_adam_learning_rate_%.1e_n_epochs_%d_run_id_%d_%f_cases_%s" % (n_rb_modes, n_layers, n_neurons_internal, learning_rate, number_of_epochs, run_id, training_ratio * 100, time_str)


  # Network training.
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=history_output, histogram_freq=0, write_graph=False,
                                                        write_images=False, update_freq='epoch', profile_batch=2,
                                                        embeddings_freq=0, embeddings_metadata=None)

  nn.fit(training_inputs, training_labels,
         epochs=number_of_epochs,
         validation_split=validation_split,
         batch_size=64,
         callbacks=[tensorboard_callback],
         # verbose=0)
         )

  nn.save(model_output)


training_ratio = 1.0
n_rb_modes = 40
n_layers = 5
n_neurons_internal = 500
learning_rate = 5.0e-3
number_of_epochs = 2000
validation_split = 0.20
train_nn(n_rb_modes, training_ratio, n_layers, n_neurons_internal, validation_split, number_of_epochs, learning_rate)
