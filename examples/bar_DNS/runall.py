
import sys, os
from itertools import product
os.environ['MKL_THREADING_LAYER'] = 'GNU' # prevent strange error MKL 

if __name__ == '__main__':
    
    main_script = 'solveDNS.py'
    main_script_mesh = 'mesh_generation_DNS.py'
    
    Ny_list = [24, 72]
    
    for Ny in Ny_list:        
        print("Running with : %d"%(Ny))
        os.system("python %s %d %s %s %d"%(main_script_mesh, Ny, 'True', 'True', -1)) # last number is dummy
        os.system("python %s %d"%(main_script, Ny))
