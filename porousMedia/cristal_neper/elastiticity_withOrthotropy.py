#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Available in: https://github.com/felipefr/micmacsFenics.git
@author: Felipe Figueredo Rocha, f.rocha.felipe@gmail.com,
felipe.figueredorocha@epfl.ch

Bar problem given a constitutive law (single-scale):
Problem in [0,Lx]x[0,Ly], homogeneous dirichlet on left and traction on the
right. We use an isotropic linear material, given two lamé parameters.
"""
import sys, os
import dolfin as df
import numpy as np
from ufl import nabla_div
sys.path.insert(0, '../../utils/')
from fenicsUtils import symgrad
from enriched_mesh import EnrichedMesh

resultFolder = './'

class myCoeff(df.UserExpression):
    def __init__(self, markers, coeffs, id_0, **kwargs):
        self.markers = markers
        self.coeffs = coeffs
        self.id_0 = id_0
        super().__init__(**kwargs)

        
    def eval_cell(self, values, x, cell):
        values[0] = self.coeffs[self.markers[cell.index] - self.id_0]



class C_orth(df.UserExpression):
    def __init__(self, markers, angles, id_0, **kwargs):
        self.markers = markers
        self.angles = angles
        self.id_0 = id_0
        super().__init__(**kwargs)

        
    def eval_cell(self, values, x, cell):
        values[0] = self.coeffs[self.markers[cell.index] - self.id_0]


Lx = 1.0
Ly = 1.0

id_cristal_0 = 5
nCristal = 14

lamb = np.linspace(3., 7., nCristal)
mu = np.linspace(1.,3.,nCristal) 



tx = 1.0

# Create mesh and define function space
mesh = EnrichedMesh('meshes/handmade_polycristal.xml')

# mesh = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(Lx, Ly),
                        # Nx, Ny, "right/left")

Uh = df.VectorFunctionSpace(mesh, "Lagrange", 1)

leftBnd = df.CompiledSubDomain('near(x[0], 0.0) && on_boundary')
rightBnd = df.CompiledSubDomain('near(x[0], Lx) && on_boundary', Lx=Lx)

boundary_markers = df.MeshFunction("size_t", mesh, dim=1, value=0)
leftBnd.mark(boundary_markers, 1)
rightBnd.mark(boundary_markers, 2)

# Define boundary condition
bcL = df.DirichletBC(Uh, df.Constant((0.0, 0.0)), boundary_markers, 1)

ds = df.Measure('ds', domain=mesh, subdomain_data=boundary_markers)
traction = df.Constant((tx, 0.0))

lamb_ = myCoeff(mesh.subdomains, lamb, id_cristal_0)
mu_ = myCoeff(mesh.subdomains, mu, id_cristal_0)


def sigma(u):
    return lamb_*nabla_div(u)*df.Identity(2) + 2*mu_*symgrad(u)


# Define variational problem
uh = df.TrialFunction(Uh)
vh = df.TestFunction(Uh)
a = sum([df.inner(sigma(uh), df.grad(vh))*mesh.dx(i + id_cristal_0) for i in range(nCristal)])
b = df.inner(traction, vh)*ds(2)

# Compute solution
uh = df.Function(Uh)

# linear_solver ops: "superlu" or "mumps"
df.solve(a == b, uh, bcs=bcL, solver_parameters={"linear_solver": "mumps"})

# Save solution in VTK format
fileResults = df.XDMFFile(resultFolder + "bar_single_scale.xdmf")
fileResults.write(uh)