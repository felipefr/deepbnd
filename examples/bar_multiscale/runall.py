"""
This file is part of deepBND, a data-driven enhanced boundary condition implementaion for 
computational homogenization problems, using RB-ROM and Neural Networks.
Copyright (c) 2020-2023, Felipe Rocha.
See file LICENSE.txt for license information. 
Please cite this work according to README.md.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or <felipe.f.rocha@gmail.com>
"""

import sys, os
from itertools import product
os.environ['MKL_THREADING_LAYER'] = 'GNU' # prevent strange error MKL 

if __name__ == '__main__':
    
    main_script = 'barMultiscale.py'
    main_script_mesh = 'mesh_generation_RVEs.py'
    
    caseType_list = ['reduced_per', 'dnn', 'full']
    createMesh_list = ['True', 'False', 'False']
    
    Ny_list = [24, 72]
    meshsize_list = ['reduced']
    
    # for Ny, meshsize in product(Ny_list, meshsize_list):        
    #     print("Running with : %d"%(Ny))
    #     os.system("python %s %d %d %d %s"%(main_script_mesh, Ny, run, num_runs, meshsize) )# last number is dummy
    #     # os.system("python %s %d"%(main_script, Ny))
        
    for Ny in Ny_list:
        for caseType, createMesh in zip(caseType_list, createMesh_list):
            
            print("Running with : %d %s"%(Ny, caseType))
            os.system("python %s %d %s %s %d"%(main_script, Ny, caseType, createMesh, 96))
