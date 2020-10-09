#!/bin/bash
# Execute this file to recompile locally
c++ -Wall -shared -fPIC -std=c++11 -fno-lto -undefined dynamic_lookup -DDOLFIN_VERSION=2019.1.0 -DNDEBUG -DHAS_HDF5 -D_FORTIFY_SOURCE=2 -DHAS_SLEPC -DHAS_PETSC -DHAS_UMFPACK -DHAS_CHOLMOD -DHAS_SCOTCH -DHAS_PARMETIS -DHAS_ZLIB -DHAS_MPI -DDOLFIN_VERSION=2019.1.0 -O3 -fno-math-errno -fno-trapping-math -ffinite-math-only -I/Users/felipefr/opt/anaconda2/envs/tf115_env/include -I/Users/felipefr/opt/anaconda2/envs/tf115_env/include/eigen3 -I/Users/felipefr/opt/anaconda2/envs/tf115_env/include/python3.6m -I/Users/felipefr/opt/anaconda2/envs/tf115_env/.cache/dijitso/include dolfin_cpp_module_0c1e91e6a2ca10d08ec570f60076fe26.cpp -L/Users/felipefr/opt/anaconda2/envs/tf115_env/lib -L/Users/felipefr/opt/anaconda2/envs/tf115_env/Users/felipefr/opt/anaconda2/envs/tf115_env/lib -L/Users/felipefr/opt/anaconda2/envs/tf115_env/.cache/dijitso/lib -Wl,-rpath,/Users/felipefr/opt/anaconda2/envs/tf115_env/.cache/dijitso/lib -lpmpi -lmpi -lmpicxx -lpetsc -lslepc -lm -ldl -lz -lpthread -lhdf5 -lboost_timer -ldolfin -Wl,-install_name,/Users/felipefr/opt/anaconda2/envs/tf115_env/.cache/dijitso/lib/dolfin_cpp_module_0c1e91e6a2ca10d08ec570f60076fe26.so -odolfin_cpp_module_0c1e91e6a2ca10d08ec570f60076fe26.so