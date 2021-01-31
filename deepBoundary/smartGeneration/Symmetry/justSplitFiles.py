import sys, os
import numpy as np
sys.path.insert(0, '../../../utils/')
# sys.path.insert(0, '../training3Nets/')

import fenicsWrapperElasticity as fela
import generation_deepBoundary_lib as gdb
from dolfin import *
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import myHDF5 as myhd
import meshUtils as meut
import fenicsUtils as feut

folder = ["/Users", "/home"][1] + "/felipefr/switchdrive/scratch/deepBoundary/smartGeneration/LHS_frozen_p4_volFraction/"
folderBasis = ["/Users", "/home"][1] + "/felipefr/switchdrive/scratch/deepBoundary/smartGeneration/LHS_frozen_p4_volFraction/"
nameSnaps = folder + 'snapshots_all.h5'
nameC = folderBasis + 'Cnew.h5'
nameMeshRefBnd = 'boundaryMesh.xdmf'
nameWbasis = folderBasis + 'Wbasis_new.h5'
nameWbasisT1 = folderBasis + 'Wbasis_T1.h5'
nameWbasisT2 = folderBasis + 'Wbasis_T2.h5'
nameWbasisT3 = folderBasis + 'Wbasis_T3.h5'
nameYlist = folder + 'Y_validation_p4.h5'
nameYlistT1 = folder + 'Y_validation_p4_T1.h5'
nameYlistT2 = folder + 'Y_validation_p4_T2.h5'
nameYlistT3 = folder + 'Y_validation_p4_T3.h5'
nameTau = folderBasis + 'tau2.h5'
nameEllipseData = folder + 'ellipseData_validation.h5'

os.system('rm test.hd5')
myhd.extractDataset(nameSnaps, 'test.hd5', 'solutions_trans', 'a')