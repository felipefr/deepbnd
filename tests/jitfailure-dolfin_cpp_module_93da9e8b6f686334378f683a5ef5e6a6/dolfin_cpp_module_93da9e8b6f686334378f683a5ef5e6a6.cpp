
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <dolfin/function/Expression.h>

typedef Eigen::VectorXd npArray;
typedef Eigen::VectorXi npArrayInt;

class myCoeff : public dolfin::Expression {
  public:
    
    npArray coeffs; // dynamic double vector
    npArrayInt materials; // dynamic integer vector
    int nc;
    
    myCoeff(npArray c, npArrayInt mat, int nc) : dolfin::Expression(nc), coeffs(c), materials(mat), nc(nc) { }

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

PYBIND11_MODULE(dolfin_cpp_module_93da9e8b6f686334378f683a5ef5e6a6, m) {
    pybind11::class_<myCoeff, std::shared_ptr<myCoeff>, dolfin::Expression>
    (m, "myCoeff")
    .def(pybind11::init<npArray,npArrayInt>())
    .def("__call__", &myCoeff::eval)
    .def("updateCoeffs", &myCoeff::updateCoeffs);
}
