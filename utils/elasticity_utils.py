import numpy as np
import myCoeffClass as coef
# import myLibRB as myrb

gof = lambda g,f: lambda x,y : g(*f(x,y)) # composition, g : R2 -> R* , f : R2 -> R2

lame2youngPoisson  = lambda lamb, mu : [ 0.5*lamb/(mu + lamb) , mu*(3.*lamb + 2.*mu)/(lamb + mu) ]
youngPoisson2lame = lambda nu,E : [ nu * E/((1. - 2.*nu)*(1.+nu)) , E/(2.*(1. + nu)) ]


lame2lameStar = lambda lamb, mu: [(2.0*mu*lamb)/(lamb + 2.0*mu), mu]
lameStar2lame = lambda lambStar, mu: [(2.0*mu*lambStar)/(-lambStar + 2.0*mu), mu]

eng2lamb = lambda nu, E: nu * E/((1. - 2.*nu)*(1.+nu))
eng2mu = lambda nu, E: E/(2.*(1. + nu))
lame2poisson = lambda lamb, mu: 0.5*lamb/(mu + lamb)
lame2young = lambda lamb, mu: mu*(3.*lamb + 2.*mu)/(lamb + mu)
lame2lambPlane = lambda lamb, mu: (2.0*mu*lamb)/(lamb + 2.0*mu)
lamePlane2lamb = lambda lambStar, mu: (2.0*mu*lambStar)/(-lambStar + 2.0*mu)

eng2mu = lambda nu, E: E/(2.*(1. + nu))
eng2lambPlane = gof(lame2lambPlane,lambda x,y: (eng2lamb(x,y), eng2mu(x,y))) 

def youngPoisson2lame_planeStress(nu,E):
    lamb , mu = youngPoisson2lame(nu,E)
    
    lamb = (2.0*mu*lamb)/(lamb + 2.0*mu)
    
    return lamb, mu

convertParam2 = lambda p,f: np.array( [  f(*p_i) for p_i in p ] )

def convertParam(param,foo):
    
    n = len(param)
    paramNew = np.zeros((n,2))
    for i in range(n):
        paramNew[i,0], paramNew[i,1] = foo( *param[i,:].tolist()) 
  
    return paramNew

      
def tensor2voigt_sym(A):
    return np.array([A[0,0],A[1,1],A[0,1] + A[1,0]])


def getLameInclusions(nu1,E1,nu2,E2,M, op='cpp'):
    mu1 = eng2mu(nu1,E1)
    lamb1 = eng2lambPlane(nu1,E1)
    mu2 = eng2mu(nu2,E2)
    lamb2 = eng2lambPlane(nu2,E2)
    param = np.array([[lamb1, mu1], [lamb2,mu2],[lamb1,mu1], [lamb2,mu2]])
    
    materials = M.subdomains.array().astype('int32')
    materials = materials - np.min(materials)
    
    lame = coef.getMyCoeff(materials, param, op = op)
    
    return lame


# Either mandel or voigt since specific entries are zero
def orthotropicElasticityTensor(Ex,Ey,vxy):
    vyx = Ex*vxy/Ey # vyx/Ex = vxy/Ey should hold
    Gxy = Ex/(2.0*(1+vxy))

    # compliance tensor
    S = np.array([[1./Ex, -vxy/Ey, 0], [-vyx/Ex, 1./Ey, 0], [0, 0, 1/Gxy]] )
    
    # elasticity tensor
    C = np.linalg.inv(S)
    
    return C


def rotationInMandelNotation(theta):

    Q = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    
    # Rotation tranformation in mandel-kelvin convention
    sq2 = np.sqrt(2.0)
    Tm = np.array([ [Q[0,0]**2 , Q[0,1]**2, sq2*Q[0,0]*Q[0,1]], 
                    [Q[1,0]**2 , Q[1,1]**2, sq2*Q[1,1]*Q[1,0]],
                    [sq2*Q[1,0]*Q[0,0] , sq2*Q[0,1]*Q[1,1], Q[1,1]*Q[0,0] + Q[0,1]*Q[1,0] ] ])
    
    
    return Tm