import dolfin as dl
mesh = dl.UnitSquareMesh(10,10)
V = dl.FunctionSpace(mesh, "CG",1)
def point0(x,on_boundary):
    return x[0] < dl.DOLFIN_EPS and x[1] < dl.DOLFIN_EPS
bc_point0 = dl.DirichletBC(V, dl.Constant(0), point0, 'pointwise')