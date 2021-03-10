import dolfin as df
import numpy as np
import meshUtils as meut

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


def affineTransformationExpession(a,B, mesh):
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
    
    return affineTransformationExpession(a,B, mesh) , a, B

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
