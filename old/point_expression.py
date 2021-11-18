import dolfin as df 
import numpy as np 

code = """
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <dolfin/function/Expression.h>
#include <iostream>
#include <math.h>

typedef Eigen::Ref<Eigen::VectorXd> npArray;
typedef Eigen::Ref<Eigen::VectorXi> npArrayInt;

class PointExpression : public dolfin::Expression {
  public:
    
    npArray u; // dynamic double vector
    double x0,x1,y0,y1; 
    int npoints;
    
    PointExpression(npArray u_, double x0_, double x1_, double y0_, double y1_, int n) : dolfin::Expression(2), 
        u(u_), x0(x0_), x1(x1_), y0(y0_), y1(y1_), npoints(n) { }

    void eval(Eigen::Ref<Eigen::VectorXd> values,
                      Eigen::Ref<const Eigen::VectorXd> x) const {
        
        double tol = 1.0e-5;
        int i,j,k;
        double s,si, omega;
                          
        if(fabs(x[1] - y0)<tol){
            s = 0.25*(x[0] - x0)/(x1 - x0);
        } else if(fabs(x[0] - x1)<tol){
            s = 0.25 + 0.25*(x[1] - y0)/(y1 - y0);
        } else if(fabs(x[1] - y1)<tol){
            s = 0.5 + 0.25*(x[0] - x1)/(x0 - x1);
        } else if(fabs(x[0] - x0)<tol){
            s = 0.75 + 0.25*(x[1] - y1)/(y0 - y1);
        } else{
            s = -1.0;
            values[0] = 0.0; values[1] = 0.0;
        }
            
        
        if (s > -0.1){
            si = s*npoints;   
            i = int(si);
            omega = si - float(i); 
            
            j = (i+1)%npoints;
            
            std :: cout << si << i << omega << j << npoints <<std :: endl;    
            
            for(k = 0; k<2; k++){
                values[k] = (1.0-omega)*u[2*i + k]  + omega*u[2*j + k]; 
            }
        }
        
        std :: cout << "val0=" << values[0] << " val1=" << values[1] << std :: endl;
    
    }
            
    void updateU(Eigen::VectorXd u_, int n){ 
        for(int i = 0; i<n; i++){
          u[i]= u_[i];
        }
    }
                    
};

PYBIND11_MODULE(SIGNATURE, m) {
    pybind11::class_<PointExpression, std::shared_ptr<PointExpression>, dolfin::Expression>
    (m, "PointExpression")
    .def(pybind11::init<npArray,double,double,double,double,int>())
    .def("__call__", &PointExpression::eval)
    .def("updateU", &PointExpression::updateU);
}
"""

def PointExpression(u,gen, op = 'python'):
    if(op == 'python'):
        return PointExpressionPython(u,gen)

    elif(op == 'cpp'):
        exprCpp = df.compile_cpp_code(code)
        
        npoints = gen.npoints
        x0 = np.min(gen.x_eval[:,0])
        x1 = np.max(gen.x_eval[:,0])
        y0 = np.min(gen.x_eval[:,1])
        y1 = np.max(gen.x_eval[:,1])        
        uu = u.flatten().astype('float64')
        n = 160
        obj = exprCpp.PointExpression(uu,x0,x1,y0,y1,npoints)
        obj.updateU(uu,n)
        return df.CompiledExpression(obj, degree = 1) 


        # return PointExpressionCpp(u,gen)
        
class PointExpressionPython(df.UserExpression):
    def __init__(self, u, gen):
        super().__init__()
        self.npoints = gen.npoints
        self.u = u.flatten()
        self.tol = 1e-5
        self.x0 = np.min(gen.x_eval[:,0])
        self.x1 = np.max(gen.x_eval[:,0])
        self.y0 = np.min(gen.x_eval[:,1])
        self.y1 = np.max(gen.x_eval[:,1])
        
    def eval(self, value, x):
        
        s = 0.0
        
        if(np.abs(x[1] - self.y0)<self.tol):
            # print(x, "in bottom")
            s = 0.25*(x[0] - self.x0)/(self.x1 - self.x0)
        elif(np.abs(x[0] - self.x1)<self.tol):
            # print(x, "on right")
            s = 0.25 + 0.25*(x[1] - self.y0)/(self.y1 - self.y0)
        elif(np.abs(x[1] - self.y1)<self.tol):
            # print(x, "on top")
            s = 0.5 + 0.25*(x[0] - self.x1)/(self.x0 - self.x1)
        elif(np.abs(x[0] - self.x0)<self.tol):
            # print(x, "on left")
            s = 0.75 + 0.25*(x[1] - self.y1)/(self.y0 - self.y1)
        else:
            s = -1.0
            value[:] = 0.0
        
        if s > -0.1:
            si = s*self.npoints   
            i = int(si)
            omega = si - float(i) 
            
            j = (i+1)%self.npoints
            
            for k in range(2):
                value[k] = (1.0-omega)*self.u[2*i + k]  + omega*self.u[2*j + k] 

    def value_shape(self):
        return (2,)
    