"""
This file is part of deepBND, a data-driven enhanced boundary condition implementaion for 
computational homogenization problems, using RB-ROM and Neural Networks.
Copyright (c) 2020-2023, Felipe Rocha.
See file LICENSE.txt for license information. 
Please cite this work according to README.md.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or <felipe.f.rocha@gmail.com>
"""

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
   
# snapFile = foldersnap + 'snapshots{0}_{1}.h5'
snapFileMerged = foldersnap + 'snapshots{0}.h5'.format(model)
# os.system('rm ' + snapFileMerged)
# myhd.merge([snapFile.format(model,i) for i in range(num_runs)], snapFileMerged, 
#             InputLabels = labels, OutputLabels = labels, axis = 0, mode = 'w-')

# [os.system('rm ' + snapFile.format(model,i)) for i in range(num_runs)]

snap_all = myhd.loadhd5(snapFileMerged,  labels)

ids = snap_all[0]
    
indexes = np.argsort(ids)

ids_sorted = ids[indexes].astype('int')

ids_sorted_unique, index_unique = np.unique(ids_sorted, axis=0, return_index=True)

new_snaps_fields = []

for f in snap_all:
    new_snaps_fields.append(f[indexes][index_unique])

snapFileMerged_selected = foldersnap + 'snapshots{0}_selected.h5'.format(model)

os.system('rm ' + snapFileMerged_selected)
myhd.savehd5(snapFileMerged_selected, new_snaps_fields, labels, 'w-')
