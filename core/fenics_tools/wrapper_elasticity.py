import dolfin as df
from ufl import nabla_div
from core.fenics_tools.misc import symgrad


halfsq2 = np.sqrt(2.)/2. # useful for mandel notation

# Voigt notation for strain (for stress becomes only sigma_12 at third entry)
symgrad_voigt = lambda v: df.as_vector([v[0].dx(0), 
                                        v[1].dx(1), 
                                        v[0].dx(1) + v[1].dx(0) ])

# Mandel-Kelvin notation for both strain and stress
symgrad_mandel = lambda v: df.as_vector([v[0].dx(0), 
                                         v[1].dx(1), 
                                         halfsq2*(v[0].dx(1) + v[1].dx(0)) ])

# REMOVE? equivalent to symgrad
def epsilon(u):
    return 0.5*(df.nabla_grad(u) + df.nabla_grad(u).T)

def sigmaLame(u, lame):
    return lame[0]*nabla_div(u)*df.Identity(2) + 2*lame[1]*symgrad(u)

def vonMises(sig):
    s = sig - (1./3)*df.tr(sig)*df.Identity(2)
    return df.sqrt((3./2)*df.inner(s, s)) 