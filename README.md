# DeepBND
A Deep Learning-based method to enhance boundary conditions. This code has been used to simulate the numerical examples of the article "DeepBND: a Machine Learning approach to enhance Multiscale Solid Mechanics" (https://arxiv.org/abs/2110.11141).



**Installation** 

Among other libraries, DeepBND relies on the following ones (some of them are installed automatically in the installation of previous ones):

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

**Testing** 
```
cd tests
pytest test_*            
```

