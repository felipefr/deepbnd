from joblib import Parallel, delayed, parallel_backend
from math import sqrt
from timeit import default_timer as timer
import numpy as np
import multiprocessing

n = 100000

start = timer()
with parallel_backend('loky', n_jobs=100):
    a = Parallel()(delayed(sqrt)(i ** 2) for i in range(100000))
# a = [sqrt(i ** 2) for i in range(n)]
# a = np.sqrt(np.arange(n)**2)
end = timer()
print(end-start)