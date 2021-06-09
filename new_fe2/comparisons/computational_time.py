import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0,'../../utils/')

# import generation_deepBoundary_lib as gdb
import matplotlib.pyplot as plt
import numpy as np

# import myTensorflow as mytf
from timeit import default_timer as timer

import h5py
import myHDF5 as myhd
import matplotlib.pyplot as plt
import meshUtils as meut
from dolfin import *

def load_times(file_name, string_to_search = "concluded in"):
    list_of_results = []
    with open(file_name, 'r') as read_obj:
        for line in read_obj:
            if string_to_search in line:
                list_of_results.append(float(line.split('  ')[-1])) 
    
    return np.array(list_of_results)

# Test Loading 

log_val_file = 'log_validation_LLHS_part{0}.txt'

list_times = []
for i in range(3):
    times = load_times(log_val_file.format(i))
    list_times.append(times)
    print(i, np.mean(times),np.std(times))


times = np.array(list_times)
print("total",  np.mean(times),np.std(times))
