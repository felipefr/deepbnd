from dolfin import *
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


mesh = UnitSquareMesh(MPI.COMM_SELF, 32, 32) # this trick make the mesh specific to the rank, and so solves to problems
V = FunctionSpace(mesh, "Lagrange", 1)
u = TrialFunction(V)
v = TestFunction(V)

if(rank==1):
    g = Expression('sin(pi*x[0])*sin(pi*x[1])',pi = np.pi, degree = 1 )
    f = Expression('2*pi*pi*sin(pi*x[0])*sin(pi*x[1])',pi = np.pi, degree = 1)
elif(rank==0):
    g = Expression('exp(-ll*(x[0]-0.5) )+exp(ll*(x[0]-0.5))',ll=20.0 , degree = 1)
    f = Expression('-ll*ll*exp(-ll*(x[0]-0.5))-ll*ll*exp(ll*(x[0]-0.5))',ll=20.0, degree = 1)
    
class Boundary(SubDomain):
    def inside(self,x,on_boundary):
        return on_boundary      

bc = DirichletBC(V, g, Boundary())
a = inner(grad(u), grad(v))*dx 
L = f*v*dx 
u = Function(V)
solve(a == L ,u, bcs = bc, solver_parameters={"linear_solver": "superlu"})

print(u(Point(0.5,0.5)))