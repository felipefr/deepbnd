import os
import shutil

resdir = 'results/'

subdirs = next(os.walk(resdir))[1]

for subd in subdirs:
    if not os.path.isfile( resdir + subd + '/errors.npy' ):
        print( resdir + subd )
        shutil.rmtree( resdir + subd )
