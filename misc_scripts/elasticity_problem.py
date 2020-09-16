import numpy as np
import math

import pyorb_core.pde_problem.fom_problem as fp

# def aa_theta_a( _param, _q ):
#     assert _q <= 2

#     nu = _param[0]
#     E = _param[1]
   
#     if( _q == 0 ):
#         mu = E/(2.*(1. + nu))
#         return 2.0*mu 
#     elif( _q == 1 ):
#         lamb = nu * E/((1. - 2.*nu)*(1.+nu))
#         return lamb 


def aa_theta_a( _param, _q ):
    assert _q <= 2

    lamb = _param[0]
    mu = _param[1]
   
    if( _q == 0 ):
        return lamb
    elif( _q == 1 ):
        return mu 


# def aa_theta_f( _param, _q ):
#     if(_q == 0):
#         return 1.0
    
def aa_theta_f( _param, _q ):
    if(_q == 0):
        return np.cos(_param[2])
    if(_q == 1):
        return np.sin(_param[2])

class elasticity_problem( fp.fom_problem ):

    def __init__( self, _parameter_handler ):
        fp.fom_problem.__init__( self, _parameter_handler )

        return

    def define_theta_functions( self ):

        self.M_theta_a = aa_theta_a
        self.M_theta_f = aa_theta_f

        return
