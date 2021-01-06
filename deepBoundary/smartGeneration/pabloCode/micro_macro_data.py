import numpy as np
import h5py

def read_raw_data(n_rb_modes):
  fname_input = 'ellipseData_1.h5'
  fname_output = 'Y.h5'

  fin = h5py.File(fname_input, 'r')
  fout = h5py.File(fname_output, 'r')

  input_data = np.array(fin['ellipseData'])
  output_data = np.array(fout['Ylist'])

  return input_data[:, :, 2], output_data[:, :n_rb_modes]

def read_micro_macro_data(n_rb_modes=40, shuffle=True, normalize_input=True, normalize_output=True, ratio_samples=1.0):

  input_data, output_data = read_raw_data(n_rb_modes)

  if shuffle:
    n = input_data.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    input_data = input_data[indices, :]
    output_data = output_data[indices]

  if normalize_input:
    for i in range(input_data.shape[1]):
      a0 = np.min(input_data[:, i])
      a1 = np.max(input_data[:, i])
      input_data[:, i] = (input_data[:, i] - a0) / (a1 - a0)

  if normalize_output:
    for i in range(output_data.shape[1]):
      a0 = np.min(output_data[:, i])
      a1 = np.max(output_data[:, i])
      output_data[:, i] = (output_data[:, i] - a0) / (a1 - a0)

  if ratio_samples < 1.0:
    n = input_data.shape[0]
    nn = int(np.floor(n * ratio_samples))
    input_data = input_data[:nn, :]
    output_data = output_data[:nn, :]

  return input_data, output_data

