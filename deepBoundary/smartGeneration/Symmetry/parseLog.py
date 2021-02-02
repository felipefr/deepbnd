import sys
import numpy as np
# namefile = sys.argv[1]
namefile = input()

loss = []
loss_val = []
with open(namefile,'r') as f:
    for line in f:
        if(line[0:9] == '1024/1024'):
            i0 = line.find('loss:') + 5
            i1 = line.find('<lambda>:') - 3
            
            j0 = line.find('val_loss:') + 9
            j1 = line.find('val_<lambda>:') - 3
            
            if(j0>8):
                loss.append(float(line[i0:i1]))
                loss_val.append(float(line[j0:j1]))

loss = np.array(loss).reshape((-1,1))
loss_val = np.array(loss_val).reshape((-1,1))
np.savetxt('loss_'+ namefile, np.concatenate((loss,loss_val), axis = 1))