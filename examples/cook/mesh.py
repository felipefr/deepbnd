"""
This file is part of deepBND, a data-driven enhanced boundary condition implementaion for 
computational homogenization problems, using RB-ROM and Neural Networks.
Copyright (c) 2020-2023, Felipe Rocha.
See file LICENSE.txt for license information. 
Please cite this work according to README.md.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or <felipe.f.rocha@gmail.com>
"""

import sys, os
import dolfin as df 
import matplotlib.pyplot as plt

from deepBND.core.mesh.wrapper_gmsh import myGmsh
import numpy as np

class CookMembrane(myGmsh):
    def __init__(self, lcar = 1.0):
        super().__init__()    
              
        Lx = 48.0
        Hy = 44.0
        hy = 16.0
        
        hsplit = int(np.sqrt(Lx**2 + Hy**2)/lcar)
        vsplit = int(0.5*(Hy + hy)/lcar)
        
        p0 = self.add_point([0.0,0.0,0.0], lcar = lcar)
        p1 = self.add_point([Lx,Hy,0.0], lcar = lcar)
        p2 = self.add_point([Lx,Hy+hy,0.0], lcar = lcar)
        p3 = self.add_point([0.0,Hy,0.0], lcar = lcar)
        
        l0 = self.add_line(p0,p1)
        l1 = self.add_line(p1,p2)
        l2 = self.add_line(p2,p3)
        l3 = self.add_line(p3,p0)
        
        self.l = [l0,l1,l2,l3]
        a = self.add_line_loop(lines = self.l)
        self.s = self.add_surface(a)
        
        self.set_transfinite_lines(self.l[0::2], hsplit)
        self.set_transfinite_lines(self.l[1::2], vsplit)
        self.set_transfinite_surface(self.s,orientation = 'alternate')

        self.physicalNaming()
        
    def physicalNaming(self):
        self.add_physical(self.l[1], 1)
        self.add_physical(self.l[3], 2)
        self.add_physical(self.s,0)
        
        
