#!/bin/bash
# Execute this file to recompile locally
c++ -Wall -shared -fPIC -std=c++11 -O2 -I/Users/felipefr/opt/anaconda2/envs/tf115_env/include -I/Users/felipefr/opt/anaconda2/envs/tf115_env/include/eigen3 -I/Users/felipefr/opt/anaconda2/envs/tf115_env/.cache/dijitso/include ffc_form_197fdfbb79303987f5ad8837d9392a1bf40b2921.cpp -L/Users/felipefr/opt/anaconda2/envs/tf115_env/.cache/dijitso/lib -Wl,-rpath,/Users/felipefr/opt/anaconda2/envs/tf115_env/.cache/dijitso/lib -ldijitso-ffc_element_03fd857de211afe809e25f7e773dba1cb2a08845 -ldijitso-ffc_element_fe0a4aa76f67b05b91fd87e746f22ff05cc31ea4 -ldijitso-ffc_coordinate_mapping_093869bc26937bb34dc6ea89858ec40896423fd9 -Wl,-install_name,/Users/felipefr/opt/anaconda2/envs/tf115_env/.cache/dijitso/lib/libdijitso-ffc_form_197fdfbb79303987f5ad8837d9392a1bf40b2921.so -olibdijitso-ffc_form_197fdfbb79303987f5ad8837d9392a1bf40b2921.so