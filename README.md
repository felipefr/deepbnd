# DeepBND
A Deep Learning-based method to enhance boundary conditions. This code has been used to simulate the numerical examples of the article "DeepBND: a Machine Learning approach to enhance Multiscale Solid Mechanics" (https://arxiv.org/abs/2110.11141).

## Installation 

Among other libraries, deepBND relies on the following ones (some of them are installed automatically in the installation of previous ones):

- library          version  
- python 		   3.8.10 (recommended) 
- tensorflow       2.5.0 (conda-forge)
- fenics           2019.1.0   (conda-forge)
- multiphenics     0.2.dev1  (pypi)
- lxml             4.6.4  (conda-forge)
- spyder           5.0.0  (conda-forge)
- scikit-optimize  0.8.1  (conda-forge)
- h5py             3.2.1  (pypi)   
- meshio           3.3.1  (pypi)
- pygmsh           6.0.2  (pypi)
- gmsh             4.6.0   (pypi)
- pytest 		   6.2.5.  (pypi)
- mkl              2021.2.0 
- mpi4py           3.0.3 
- hdf5             1.10.6 
- micmacsfenics    (https://github.com/felipefr/micmacsfenics.git)

Obs: the default repository is conda-forge, otherwise pypi from pip. Recommended versions should be understood only as guideline and sometimes the very same version is not indeed mandatory. 


We recommend the use of miniconda (https://docs.conda.io/en/latest/miniconda.html)

- To create a fresh environment with fenics and tensorflow:
```
conda create -n tf-fenics -c conda-forge python=3.8.10 fenics=2019.1 tensorflow=2.5
```

- To activate the environment:
```
conda activate tf-fenics
```

- To install additional packages:
```
conda install -c conda-forge <name_of_the_package>=<version>
```
or use pip
```
pip install <name_of_the_package>==<version>
```

or at once (which was tested)

```
conda install -c conda-forge scikit-optimize h5py lxml spyder
pip install pytest gmsh==4.6.0 meshio==3.3.1 pygmsh==6.0.2
git clone https://github.com/multiphenics/multiphenics.git
cd multiphenics
python3 setup.py install
```

- Intalling multiphenics:
```
git clone https://github.com/multiphenics/multiphenics.git
cd multiphenics
python3 setup.py install
```
- Installing micmacsfenics (no installation at the moment, just make the library can be retrieved by PYTHONPATH):
```
git clone https://github.com/felipefr/micmacsfenics.git
```

- Make sure your PYTHONPATH variable contains the root directory in which you cloned deepBND. By default, the anaconda installation does not take into consideration the 
OS path. You can add a .pth (any name) file listing the directories into ~/miniconda/envs/tf-fenics/lib/python3.8/site-packages. Also, you can the directories you want 
into spyder (Tools > PYTHONPATH), if you are using it.  

## Testing 
```
cd tests
pytest test_*    
or pytest --capture=no test_file.py::test_specific_test  (for detailed and specific test)      
```

## Usage

### Files
- paramRVEdataset.hd5 : It contains geometrical description for each snapshot. It allows the mesh generation, etc.  
- snapshots.hd5: simulation results for each snapshot. Here we only collect the degree of freedom on the internal boundary. For each snapshot axial and shear loads are simulated together (label 'A' and 'S', respectively). This is due to perfomance reasons (same mesh generation, reused LU decomposition, etc). 
- Wbasis.hd5: Reduced-Basis matrices and singular values. Same comment for labels 'A' and 'S' (above) applies.  
- XY.hd5: Assemble important data of paramRVEdataset.hd5 (to be used as the input of NN training) and also the projected solutions of the snapshots (snapshots.hd5) onto the RB (Wbasis.hd5). Same comment for labels 'A' and 'S' (above) applies.
- model_weights.hd5: Storages the trained weights of NN models. 
- scaler.txt: Storages the ranges used to reescale inputs and outputs of XY.hd5 ( to the range [0,1]). 
- bcs.hd5: Stores the predicted boundary conditions for a given NN model for a number of snapshots. 
- tangents.hd5: Stores the homogenised tangents (using some bc model, e.g, dnn, periodic, etc) for a number of snapshots.   

### Offline workflow

1. build_snapshots_param.py: Out = paramRVEdataset.hd5
2. simulation_snapshots.py: In = paramRVEdataset.hd5; Out = snapshots.hd5
3. build_inout.py: In = paramRVEdataset.hd5, snapshots.hd5; Out: XY.hd5, Wbasis.hd5
4. NN_training.py: In: XY.hd5 ; Out: model_weights.hd5, scaler.txt

Obs:
- Steps 1-3 should be repeated for each dataset built. Usually it consists in training, validation and testing.
- Step 4 is run with the training and validation datasets. Models should be trained indepedently for axial and shear loads ('A' and 'S' labels). 
- Additional suffix indentificators are omitted in the file names.
- Wbasis.hd5 should be obtained to the larger dataset (usually the training one), then should be reused to obtain XY.hd5 files of the remaining datasets. 
- Idem for scaler.txt.

### Prediction workflow

In the moment the prediction is offline for perfomance reasons but it can be adapted to the online workflow:

1. bc_prediction.py: In: paramRVEdataset.hd5, Wbasis.hd5, scaler.txt; Out: bcs.hd5
2. tangents_prediction.py : In: bcs.hd5, paramRVEdataset.hd5; Out: tangents.hd5

Obs:
- Step 2 is normally also performed for classical boundary conditions for comparison purposes.

### Online

#### Cook membrane example
Run cook.py with:
1. the suitable tangents.hd5 file (for the chosen bc model (caseType) ) 
2. macroscale mesh refinement indicator (Ny), 
3. A random seed to indicate how tangents.hd5 file will be shuffled (cell indexes will be matched randonmly to the RVE indexes of tangent.hd5).

Obs: You can run runall.py to automatically run several cases, this has not been extensively tested though. Hence it should be taken mostly as guide. 
#### bar_DNS (for comparison with multiscale solutions) 
1. Run mesh_generation_DNS.py with:
- Number of inclusions in the vertical direction (Ny).
- Indicate if a new param_DNS.hd5 (similar to paramRVEdataset.hd5, but to the macroscale problem, thus one single configuration instead of multiples) will sampled or reused (readReuse).
- Indicate if you want to generate paramRVEdataset.hd5 from the DNS parameters just sampled (export_paramRVE_fromDNS). It will be used in bar_multiscale. 

2. Run solveDNS.py with: 
- Enter mesh, output files, Ny and some other suffixes for identification of files purposes (as used in the mesh generation). 

Obs: You can run runall.py to automatically run several cases, this has not been extensively tested though. Hence it should be taken mostly as guide. 

#### bar_multiscale
1. (non mandatory) Run mesh_generation_RVEs.py, which create the associated meshes to paramRVEdataset.hd5 generated by mesh_generation_DNS.py, with:
- The meshsize ('reduced' or 'full').
- Other paths for meshes, input files and so. 
Obs: This script is intended to generate RVE meshes that will be used online or in tangent_predictions.py (to generate tangents.hd5 file). The later script can also generate the meshes on the fly (but of course will be slower). 

2. Run barMultiscale.py 
-  Ny_DNS: Same as Ny in the mesh_generation_DNS.py. Needed only for annotations of files purposes.  
-  caseType: Also for annotation purposes, but for the tangent file name. Indicates the multiscale model, mesh size, etc, used, e.g. per, full, reduced_per, dnn, dnn_small, etc. 
-  createMesh: Creates or reused the multiscale (coarse) mesh.
-  Ny_split_mult: If the flag createMesh is true, it is the number elements along the y axis for a regular mesh. 


Obs: You can run runall.py to automatically run several cases, this has not been extensively tested though. Hence it should be taken mostly as guide. 
## Main TODOs

- improve NatArch class, which will simplify scripts of training. 
- compatibilise micmacsfenics version of MicroconstitutiveModelDNN, in other of not repeating code (as today).
- add more tests.
- update plots and comparison scripts.



