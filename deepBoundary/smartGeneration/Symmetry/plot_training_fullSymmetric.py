import matplotlib.pyplot as plt
import numpy as np


f = open("../../../../rootDataPath.txt")
rootData = f.read()[:-1]
f.close()


folder = rootData + '/deepBoundary/smartGeneration/newTrainingSymmetry/fullSymmetric/logs_losses/'

nYlist= [5,10,15,20,25,30,35,40,60,80,100,120,140,160]

history = []

for ny in nYlist:
    history.append(np.loadtxt(folder + 'loss_log_ny{0}.txt'.format(ny)))


# for i,ny in enumerate(nYlist):
#     plt.figure(i)
#     plt.title('Optimatisation for ny = {0}'.format(ny))
#     plt.plot(history[i][:,0] , label = 'train')
#     plt.plot(history[i][:,1] , label = 'validation')
#     plt.grid()
#     plt.yscale('log')
#     plt.xlabel('epochs')
#     plt.ylabel('loss')
#     plt.legend()
#     plt.savefig(folder + 'loss_ny{0}.png'.format(ny))


plt.figure(0)
plt.title('Optimatisation all')
for i,ny in enumerate(nYlist[0::2]):
    plt.plot(history[i][:,0] , label = 'train ny ={0}'.format(ny))
    plt.grid()
    plt.yscale('log')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid()
    plt.legend()

plt.savefig(folder + 'loss_all.png')