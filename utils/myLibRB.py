import sys

sys.path.insert(0, '~/Dropbox/Luca_branch/pyorb/pyorb_core/')

import pyorb_core.rb_library.affine_decomposition as ad
import pyorb_core.rb_library.rb_manager as rm
import numpy as np

def getRBmanager(num_affine_components_A, num_affine_components_f, base_offline_folder):
    

    # ### defining the affine decomposition structure
    my_affine_decomposition = ad.AffineDecompositionHandler( )
    my_affine_decomposition.set_Q( num_affine_components_A, num_affine_components_f )    # number of affine terms
    my_affine_decomposition.import_rb_affine_matrices( base_offline_folder + 'ANq' )
    my_affine_decomposition.import_rb_affine_vectors(  base_offline_folder + 'fNq' )
    
    # ### building the RB manager
    my_rb_manager = rm.RbManager(_affine_decomposition = my_affine_decomposition, _fom_problem = None)  
    
    my_rb_manager.M_N = len(my_affine_decomposition.M_rbAffineFq[0]) 
    my_rb_manager.M_basis = np.loadtxt(base_offline_folder + 'U.txt')[:,0:my_rb_manager.M_N]
    
    return my_rb_manager