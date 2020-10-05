import matplotlib.pyplot as plt
import numpy as np

plt.rc("text", usetex = True)
plt.rc("font", family = 'serif')

def plotMeanAndStd(x, y, l='', linetypes = ['-o','--','--'], axis = 0):
    plt.plot(x, np.mean(y, axis = axis), linetypes[0], label = l)
    plt.plot(x, np.mean(y, axis = axis) + np.std(y, axis = axis) , linetypes[1], label = l + ' + std')
    plt.plot(x, np.mean(y, axis = axis) - np.std(y, axis = axis) , linetypes[2], label = l + ' - std')
    
def plotMeanAndStd_noStdLegend(x, y, l='', linetypes = ['-o','--','--'], axis = 0):
    plt.plot(x, np.mean(y, axis = axis), linetypes[0], label = l)
    plt.plot(x, np.mean(y, axis = axis) + np.std(y, axis = axis) , linetypes[1])
    plt.plot(x, np.mean(y, axis = axis) - np.std(y, axis = axis) , linetypes[2])