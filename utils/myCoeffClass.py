from dolfin import compile_cpp_code, UserExpression, CompiledExpression

code = """
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <dolfin/function/Expression.h>

typedef Eigen::Ref<Eigen::VectorXd> npArray;
typedef Eigen::Ref<Eigen::VectorXi> npArrayInt;

class myCoeff : public dolfin::Expression {
  public:
    
    npArray coeffs; // dynamic double vector
    npArrayInt materials; // dynamic integer vector
    
    myCoeff(npArray c, npArrayInt mat) : dolfin::Expression(), coeffs(c), materials(mat) { }

    void eval(Eigen::Ref<Eigen::VectorXd> values,
                      Eigen::Ref<const Eigen::VectorXd> x,
                      const ufc::cell& cell) const {
        
        values[0] = coeffs[materials[cell.index]];
    }
    
   void updateCoeffs(Eigen::VectorXd c, int n){ 
       for(int i = 0; i<n; i++){
          coeffs[i]= c[i];
       }
   }
                      
                    
};

PYBIND11_MODULE(SIGNATURE, m) {
    pybind11::class_<myCoeff, std::shared_ptr<myCoeff>, dolfin::Expression>
    (m, "myCoeff")
    .def(pybind11::init<npArray,npArrayInt>())
    .def("__call__", &myCoeff::eval)
    .def("updateCoeffs", &myCoeff::updateCoeffs);
}
"""

class myCoeff(UserExpression):
    def __init__(self, markers, coeffs, **kwargs):
        self.markers = markers
        self.coeffs = coeffs
        self.ncoeff = coeffs.shape[1]
        super().__init__(**kwargs)

        
    def eval_cell(self, values, x, cell):
        values[:self.ncoeff] = self.coeffs[self.markers[cell.index]]
        
    def value_shape(self):
        return (self.ncoeff,)


# def getMyCoeff(materials, param, op = 'cpp'):
    
#     if(op == 'cpp'):
#         myCoeffCpp = compile_cpp_code(code)
#         lamb_ = param[:,0].astype('float64')
#         mu_ = param[:,1].astype('float64')
#         lamb = CompiledExpression(myCoeffCpp.myCoeff(lamb_,materials), degree = 0)
#         mu = CompiledExpression(myCoeffCpp.myCoeff(mu_,materials), degree = 0)
#         return [lamb, mu] 

#     elif(op == 'python'):
#         return myCoeff(materials, param, degree = 0)
    
 
    
class myCoeffCpp:
    def __init__(self, materials, param):
        self.myCoeffCpp = compile_cpp_code(code)
        
        lamb_ = param[:,0].astype('float64')
        mu_ = param[:,1].astype('float64')
       
        lambHandler = self.myCoeffCpp.myCoeff(lamb_,materials)
        muHandler = self.myCoeffCpp.myCoeff(mu_,materials)
        
        self.lamb = CompiledExpression(lambHandler, degree = 0)
        self.mu = CompiledExpression(muHandler, degree = 0)
        self.values = [self.lamb,self.mu]
        
    def updateCoeffs(self, param):
        self.lamb.updateCoeffs(param[:,0].astype('float64'),param.shape[0])    
        self.mu.updateCoeffs(param[:,1].astype('float64'), param.shape[0])
        
    def __getitem__(self, key):
        print(key)
        return self.values[key]


            
def getMyCoeff(materials, param, op = 'cpp'): 
    if(op == 'cpp'):
        return myCoeffCpp(materials,param)
    elif(op == 'python'):
        return myCoeff(materials, param, degree = 2) # it was 0 before


    