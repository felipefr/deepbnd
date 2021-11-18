import numpy as np

theta = np.pi/4
Ex = 100.0
Ey = 50.0
vxy = 0.3
vyx = Ex*vxy/Ey # vyx/Ex = vxy/Ey should hold
Gxy = Ex/(2.0*(1+vxy))

Q = np.array([[np.cos(theta), np.sin(theta)],
              [-np.sin(theta), np.cos(theta)]])

# Rotation tranformation in mandel-kelvin convention
sq2 = np.sqrt(2.0)
Tm = np.array([ [Q[0,0]**2 , Q[0,1]**2, sq2*Q[0,0]*Q[0,1]], 
                [Q[1,0]**2 , Q[1,1]**2, sq2*Q[1,1]*Q[1,0]],
                [sq2*Q[1,0]*Q[0,0] , sq2*Q[0,1]*Q[1,1], Q[1,1]*Q[0,0] + Q[0,1]*Q[1,0] ] ])


Sm = np.array([[1./Ex, -vxy/Ey, 0], [-vyx/Ex, 1./Ey, 0], [0, 0, 1/Gxy]] )
Cm = np.linalg.inv(Sm)

Cm_ = Tm@Cm@Tm.T

print(Cm)
print(Cm_)

# print(np.linalg.inv(Tm))
print(Tm@Tm.T)