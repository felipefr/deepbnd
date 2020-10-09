
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <dolfin/function/Expression.h>

typedef Eigen::Ref<Eigen::VectorXd> npArray;
typedef Eigen::Ref<Eigen::VectorXi> npArrayInt;

class myCoeff2 : public dolfin::Expression {
  public:
    
    npArray coeffs; // dynamic double vector
    npArrayInt materials; // dynamic integer vector
    int ncoeff;
    
    myCoeff2(npArray c, npArrayInt mat, int nc) : dolfin::Expression(), coeffs(c), materials(mat), ncoeff(nc) { }

    void eval(Eigen::Ref<Eigen::VectorXd> values,
                      Eigen::Ref<const Eigen::VectorXd> x,
                      const ufc::cell& cell) const {
        
        for(int i = 0; i< ncoeffs; i++){
                values[i] = coeffs[ncoeffs*materials[cell.index] + i];
        }
    }
    
   void updateCoeffs(Eigen::VectorXd c, int n){ 
       for(int i = 0; i<n; i++){
          coeffs[i]= c[i];
       }
   }
                      
                    
};

PYBIND11_MODULE(dolfin_cpp_module_9c651ab0645b0bc9a8c5437e354cb873, m) {
    pybind11::class_<myCoeff2, std::shared_ptr<myCoeff2>, dolfin::Expression>
    (m, "myCoeff2")
    .def(pybind11::init<npArray,npArrayInt, int>())
    .def("__call__", &myCoeff2::eval)
    .def("updateCoeffs", &myCoeff2::updateCoeffs);
}
