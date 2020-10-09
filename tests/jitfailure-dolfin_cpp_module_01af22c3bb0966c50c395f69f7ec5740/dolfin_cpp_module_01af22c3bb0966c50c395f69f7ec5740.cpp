
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <dolfin/function/Expression.h>

typedef Eigen::Ref<Eigen::VectorXd> npArray;
typedef Eigen::Ref<Eigen::VectorXi> npArrayInt;

class myCoeff : public dolfin::Expression {
  public:
    
    npArray coeffs; // dynamic double vector
    npArrayInt materials; // dynamic integer vector
    int ncoeffs;
    
    myCoeff(npArray c, npArrayInt mat, int ncoeffs) : dolfin::Expression(), coeffs(c), materials(mat), ncoeffs(nc) { }

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

PYBIND11_MODULE(dolfin_cpp_module_01af22c3bb0966c50c395f69f7ec5740, m) {
    pybind11::class_<myCoeff, std::shared_ptr<myCoeff>, dolfin::Expression>
    (m, "myCoeff")
    .def(pybind11::init<npArray,npArrayInt>())
    .def("__call__", &myCoeff::eval)
    .def("updateCoeffs", &myCoeff::updateCoeffs);
}
