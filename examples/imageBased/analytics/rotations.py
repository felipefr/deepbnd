from sympy import simplify, Matrix, S, diff, symbols, zeros, eye
from sympy import sin, sinh, cos, cosh, sqrt
import sympy as sp

sp.init_printing(use_latex=False)


Q = sp.MatrixSymbol('Q', 2, 2)
eps = sp.MatrixSymbol('eps', 2, 2) 
epsS = (eps + eps.T)/2

eps_ = Q*epsS*Q.T

eps_m = Matrix([eps_[0,0], eps_[1,1], sqrt(2)*(eps_[0,1])])

