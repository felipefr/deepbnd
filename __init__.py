"""
This file is part of deepBND, a data-driven enhanced boundary condition implementaion for 
computational homogenization problems, using RB-ROM and Neural Networks.
Copyright (c) 2020-2023, Felipe Rocha.
See file LICENSE.txt for license information. 
Please cite this work according to README.md.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or <felipe.f.rocha@gmail.com>
"""

import os

rootPath = os.path.dirname(os.path.abspath(__file__))

HOME = os.path.expanduser('~')

rootDataPath = HOME + '/deepBND/DATA'

__all__ = ['rootDataPath']
