
import sys, os
from itertools import product

if __name__ == '__main__':
    
    main_script = 'cook.py'
    
    Ny_split_list = [5, 10, 20, 40]
    caseType_list = ['reduced_per', 'per', 'dnn', 'dnn_big', 'dnn_big_old', 'full', 'lin']
    seed_list = [0,1,2]
    createMesh_list = ['True', 'False', 'False', 'False', 'False', 'False', 'False']
    
    # ========== dataset folders ================= 
    
    for Ny_split, seed in product(Ny_split_list, seed_list):
        for caseType, createMesh in zip(caseType_list, createMesh_list):
            
            print("Running with : %d %s %d"%(Ny_split, caseType, seed))
            os.system("python %s %d %s %d %s"%(main_script, Ny_split, caseType, seed, createMesh))
