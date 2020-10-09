
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <dolfin/function/Expression.h>

typedef Eigen::<Eigen::VectorXd> npArray;
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

PYBIND11_MODULE(dolfin_cpp_module_9d28af9e83966fa8a301b7740a410393, m) {
    pybind11::class_<myCoeff, std::shared_ptr<myCoeff>, dolfin::Expression>
    (m, "myCoeff")
    .def(pybind11::init<npArray,npArrayInt>())
    .def("__call__", &myCoeff::eval)
    .def("updateCoeffs", &myCoeff::updateCoeffs);
}
