import numpy as np
import matplotlib.pyplot as plt

lame2youngPoisson  = lambda lamb, mu : [ 0.5*lamb/(mu + lamb) , mu*(3.*lamb + 2.*mu)/(lamb + mu) ]
youngPoisson2lame = lambda nu,E : [ nu * E/((1. - 2.*nu)*(1.+nu)) , E/(2.*(1. + nu)) ]

def youngPoisson2lame_planeStress(nu,E):
    lamb , mu = youngPoisson2lame(nu,E)
    
    lamb = (2.0*mu*lamb)/(lamb + 2.0*mu)
    
    return lamb, mu

def convertParam(param,foo):
    
    n = len(param)
    paramNew = np.zeros((n,2))
    for i in range(n):
        paramNew[i,0], paramNew[i,1] = foo(param[i,0],param[i,1]) 
  
    return paramNew    



nparam = 3
nsTotal = 1000

angleGravity_min = -0.2*np.pi
angleGravity_max = 0.2*np.pi

nu_min = 0.20
nu_max = 0.40

E_min = 40.0
E_max = 50.0

paramLimits = np.array([[nu_min,nu_max],[E_min,E_max],[angleGravity_min,angleGravity_max]])

param = np.array([ [ paramLimits[j,0] + np.random.uniform()*(paramLimits[j,1] - paramLimits[j,0])   for j in range(nparam)]   for i in range(nsTotal)])


plt.figure(1)

color = []

for i in range(nsTotal):
    color.append( param[i,0] ) 

plt.scatter(param[:,0],param[:,1],c = color )
plt.xlabel('poisson')
plt.ylabel('young')

plt.savefig('youngVSpoisson.png')


param[:,0:2] = convertParam(param[:,0:2], youngPoisson2lame_planeStress)
plt.figure(2)


plt.scatter(param[:,0],param[:,1], c = color)
plt.xlabel('lambda*')
plt.ylabel('mu')

plt.savefig('muVSlamb.png')

plt.show()