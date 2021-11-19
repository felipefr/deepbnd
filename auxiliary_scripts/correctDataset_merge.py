import sys, os

# sys.path.insert(0, '/home/felipefr/github/micmacsFenics/utils')
# sys.path.insert(0,'/home/rocha/source/MLELAS/utils')
sys.path.insert(0,'/home/felipefr/EPFL/newDLPDEs/MLELAS/utils')

import numpy as np
import myHDF5 as myhd
from timeit import default_timer as timer

foldersnap = './'
model = '_validation'

labels = ['id', 'solutions_S','sigma_S','a_S','B_S', 'sigmaTotal_S',
 'solutions_A','sigma_A','a_A','B_A', 'sigmaTotal_A']
    
num_runs = 2
   
snapFile = foldersnap + 'snapshots{0}_{1}.h5'
snapFileMerged = foldersnap + 'snapshots{0}.h5'.format(model)
os.system('rm ' + snapFileMerged)
myhd.merge([snapFile.format(model,i) for i in range(num_runs)], snapFileMerged, 
            InputLabels = labels, OutputLabels = labels, axis = 0, mode = 'w-')

[os.system('rm ' + snapFile.format(model,i)) for i in range(num_runs)]



snap_all = myhd.loadhd5(snapFileMerged,  labels)

ids = snap_all[0]

ids_true = np.zeros(len(ids))

k = 0
for i in range(num_runs):
    a = np.arange(i,len(ids),num_runs)
    ids_true[k:k+len(a)] = a
    k = k + len(a)
    
indexes = np.argsort(ids)

ids_sorted = ids[indexes].astype('int')

ids_sorted[0] = ids_sorted[-1] + 1 # just to not be zero

indexes_new = indexes[ids_sorted>0]

new_snaps_fields = []

for f in snap_all:
    new_snaps_fields.append(f[indexes_new])

other_ids = (ids_true - ids).astype('int')
other_ids = other_ids[other_ids>0]

snapFileMerged_selected = foldersnap + 'snapshots{0}_selected.h5'.format(model)

os.system('rm ' + snapFileMerged_selected)
myhd.savehd5(snapFileMerged_selected, new_snaps_fields, labels, 'w-')

np.savetxt("other_ids.txt", other_ids)