import matplotlib.pyplot as plt
import numpy as np

def plotMeanAndStd(x, y, a, l, linetypes = ['-o','--','--']):
    plt.plot(x, np.mean(y, axis = a), linetypes[0], label = l)
    plt.plot(x, np.mean(y, axis = a) + np.std(y, axis = a) , linetypes[1], label = l + ' + std')
    plt.plot(x, np.mean(y, axis = a) - np.std(y, axis = a) , linetypes[2], label = l + ' - std')