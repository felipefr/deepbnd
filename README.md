# DeepBND
A Deep Learning-based method to enhance boundary conditions. This code has been used to simulate the numerical examples of the article "DeepBND: a Machine Learning approach to enhance Multiscale Solid Mechanics" (https://arxiv.org/abs/2110.11141).

## Installation 

Among other libraries, deepBND relies on the following ones (some of them are installed automatically in the installation of previous ones):

- library          version  
- fenics           2019.1.0 
- multiphenics     0.2.dev1 
- h5py             3.2.1 
- hdf5             1.10.6 
- meshio           3.3.1 
- tensorflow       2.5.0 
- pygmsh           6.0.2 
- gmsh             4.6.0 
- scikit-optimize  0.8.1 
- mkl              2021.2.0 
- mpi4py           3.0.3 
- matplotlib       3.4.1 
- python           3.8.10 
- lxml             4.6.4 
- spyder           5.0.0 (recommended)     

Obs: the default repository is conda-forge, otherwise pypi from pip. Recommended versions should be understood only as guideline and sometimes the same version is not strictly necessary . 


We recommend the use of miniconda (https://docs.conda.io/en/latest/miniconda.html)

- To create a fresh environment with fenics and tensorflow:
```
conda create -n tf-fenics fenics=2019.1 tensorflow=2.5
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
You should run cook.py with:
1. the suitable tangents.hd5 file (for the chosen bc model (caseType) ) 
2. macroscale mesh refinement indicator (Ny), 
3. A random seed to indicate how tangents.hd5 file will be shuffled (cell indexes will be matched randonmly to the RVE indexes of tangent.hd5).

#### bar_multiscale
todo

#### bar_DNS (for comparison with multiscale solutions) 
todo ...




