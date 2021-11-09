import dolfin as df
import numpy as np
from timeit import default_timer as timer

def local_project(v,V):
    M = V.mesh()
    dv = df.TrialFunction(V)
    v_ = df.TestFunction(V)
    dx = df.Measure('dx', M)
    a_proj = df.inner(dv,v_)*dx 
    b_proj = df.inner(v,v_)*dx
    solver = df.LocalSolver(a_proj,b_proj) 
    solver.factorize()
    u = df.Function(V)
    solver.solve_local_rhs(u)
    return u


halfsq2 = np.sqrt(2.)/2.

symgrad = lambda v: 0.5*(df.grad(v) + df.grad(v).T)
symgrad_voigt = lambda v: df.as_vector([v[0].dx(0), v[1].dx(1), v[0].dx(1) + v[1].dx(0) ])
symgrad_mandel = lambda v: df.as_vector([v[0].dx(0), v[1].dx(1), halfsq2*(v[0].dx(1) + v[1].dx(0)) ])

def Integral(u,dx,shape):
    
    n = len(shape)
    I = np.zeros(shape)
    
    if(type(dx) != type([])):
        dx = [dx]
 
    if(n == 1):
        for i in range(shape[0]):
            for dxj in dx:
                I[i] += df.assemble(u[i]*dxj)
            
    elif(n == 2):
        for i in range(shape[0]):
            for j in range(shape[1]):
                for dxk in dx:
                    I[i,j] += df.assemble(u[i,j]*dxk)
    
    else:
        print('not implement for higher order integral')
        
    
    return I

getMesh = lambda a,b,c,d : meut.getMesh(meshGMSH = a, file = c + b + '.xml', create = d)


def affineTransformationExpression(a,B, mesh):
    return df.Expression(('a0 + B00*x[0] + B01*x[1]','a1 + B10*x[0] + B11*x[1]'), a0 = a[0], a1 = a[1],
               B00=B[0,0], B01 = B[0,1], B10 = B[1,0], B11= B[1,1] ,degree = 1, domain = mesh)

    
def getAffineTransformationLocal(U,mesh, domains_id = [], justTranslation = False):
    dxList = [mesh.dx(i) for i in domains_id]
    omegaL = sum([df.assemble(df.Constant(1.0)*dxi) for dxi in dxList])
    
    yG = Integral(df.Expression(('x[0]','x[1]'), degree = 1) , dxList, (2,))/omegaL

    uFlucL = Integral(U, dxList, (2,))/omegaL
    epsFlucL = Integral(df.grad(U), dxList, (2,2))/omegaL
    
    a = -uFlucL + epsFlucL@yG 
    B = np.zeros((2,2)) if justTranslation else -epsFlucL
    
    return affineTransformationExpression(a,B, mesh) , a, B

def getAffineTransformationLocal_bndIntegral(U,mesh, domains_id = [], justTranslation = False):
    omegaL = df.assemble(df.Constant(1.0)*mesh.dx)
    
    yG = Integral(df.Expression(('x[0]','x[1]'), degree = 1) , mesh.dx, (2,))/omegaL

    n = df.FacetNormal(mesh)
    uFlucL = Integral(U, mesh.dx, (2,))/omegaL
    epsFlucL = Integral(df.outer(U,n), mesh.ds, (2,2))/omegaL
    
    a = -uFlucL + epsFlucL@yG 
    B = np.zeros((2,2)) if justTranslation else -epsFlucL
    
    return affineTransformationExpression(a,B, mesh) , a, B

class myfog(df.UserExpression): # fog f,g : R2 -> R2, generalise 
    def __init__(self, f, g, **kwargs):
        self.f = f 
        self.g = g
        f.set_allow_extrapolation(True)
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[:2] = self.f(self.g(x))
        
    def value_shape(self):
        return (2,)

class myfog_expression(df.UserExpression): # fog f,g : R2 -> R2, generalise 
    def __init__(self, f, g, **kwargs):
        self.f = f 
        self.g = g
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[:2] = self.f(self.g(x))
        
    def value_shape(self):
        return (2,)


# code = '''
# #include <pybind11/pybind11.h>
# #include <pybind11/eigen.h>
# #include <dolfin/function/Expression.h>

# class MyFunc : public dolfin::Expression
# {
# public:
#   boost::shared_ptr<dolfin::Expression> f;
#   boost::shared_ptr<dolfin::Expression> g;

#   MyFunc() : Expression() { }

#   void eval(Array<double>& values, const Array<double>& x,
#             const ufc::cell& c) const
#   {
#       Array<double> val(2);
#       g->eval(val, x, c);
#       f->eval(values, val, c);
#   }
# };
    
# PYBIND11_MODULE(SIGNATURE, m) {
#     pybind11::class_<MyFunc, std::shared_ptr<MyFunc>, dolfin::Expression>
#     (m, "MyFunc")
#     .def("__call__", &MyFunc::eval)
# }
# '''


# code = """
# #include <pybind11/pybind11.h>
# #include <pybind11/eigen.h>
# #include <dolfin/function/Expression.h>

# typedef Eigen::Ref<Eigen::VectorXd> npArray;
# typedef Eigen::Ref<Eigen::VectorXi> npArrayInt;

# class myCoeff : public dolfin::Expression {
#   public:
    
#     npArray coeffs; // dynamic double vector
#     npArrayInt materials; // dynamic integer vector
    
#     myCoeff(npArray c, npArrayInt mat) : dolfin::Expression(), coeffs(c), materials(mat) { }

#     void eval(Eigen::Ref<Eigen::VectorXd> values,
#                       Eigen::Ref<const Eigen::VectorXd> x,
#                       const ufc::cell& cell) const {
        
#         values[0] = coeffs[materials[cell.index]];
#     }
    
#    void updateCoeffs(Eigen::VectorXd c, int n){ 
#        for(int i = 0; i<n; i++){
#           coeffs[i]= c[i];
#        }
#    }
                      
                    
# };

# PYBIND11_MODULE(SIGNATURE, m) {
#     pybind11::class_<myCoeff, std::shared_ptr<myCoeff>, dolfin::Expression>
#     (m, "myCoeff")
#     .def(pybind11::init<npArray,npArrayInt>())
#     .def("__call__", &myCoeff::eval)
#     .def("updateCoeffs", &myCoeff::updateCoeffs);
# }
# """

# def my_fog(f,g): # f(g(x))
#     compiledCode = df.compile_cpp_code(code)
#     fg = df.CompiledExpression(compiledCode, degree = 1)
#     fg.f = f
#     fg.g = g
#     return fg


def solve_iterative(a,b, bcs, Uh):
    uh = df.Function(Uh)
    
    print(Uh.dim())
    
    # solver.solve()
    start = timer()
    A, F = df.assemble_system(a, b, bcs)
    end = timer()
    print("time assembling ", end - start)
    
    solver = df.PETScKrylovSolver('gmres','hypre_amg')
    solver.parameters["relative_tolerance"] = 1e-5
    solver.parameters["absolute_tolerance"] = 1e-6
    # solver.parameters["nonzero_initial_guess"] = True
    solver.parameters["error_on_nonconvergence"] = False
    solver.parameters["maximum_iterations"] = 1000
    solver.parameters["monitor_convergence"] = True
    # solver.parameters["report"] = True
    # solver.parameters["preconditioner"]["ilu"]["fill_level"] = 1 # 
    solver.set_operator(A)
    solver.solve(uh.vector(), F)   

    return uh


def solve_direct(a,b, bcs, Uh):
    uh = df.Function(Uh)
    df.solve(a == b,uh, bcs = bcs, solver_parameters={"linear_solver": "superlu"})

    return uh
