import sys
sys.path.insert(0, '../utils/')

import matplotlib.pyplot as plt
import numpy as np
import multiphenics as mp
import dolfin as df
import meshUtils as meut
import elasticity_utils as elut
import myCoeffClass as coef
import copy
from timeit import default_timer as timer
import generationInclusions as geni
 

maxOffset = 8
offSetConstant = 1

Lx = Ly = 1.0
NxL = NyL = 2
Nx = (NxL+2*maxOffset)
Ny = (NyL+2*maxOffset)
NL = NxL*NyL
x0L = y0L = maxOffset*Lx/Nx
LxL = LyL = NxL*(x0L/maxOffset)
r0 = 0.2*LxL/NxL
r1 = 0.4*LxL/NxL
rm = np.sqrt(r0*r1)
H = Lx/Nx
Vfrac = np.pi*(rm/H)**2
xG = np.array([0.5*Lx,0.5*Ly])



def getMoments(data,xG):
    n = len(data) # number of balls
    m = np.zeros(6)
    m[0] = np.sum(data[:,2]**2)  
    for i in range(n):
        m[1:3] += data[i,2]**2 * ( data[i,0:2] - xG)
        m[3:] += data[i,2]**2 * ( np.outer(data[i,0:2] - xG,data[i,0:2] - xG) + 0.25*(data[i,2]**2)*np.eye(2)).flatten()[[0,1,3]]
    
    return m


ellipseData, permTotal, permBox = geni.circularRegular2Regions(r0, r1, NxL, NyL, Lx, Ly, maxOffset, ordered = False)
ellipseData = ellipseData[permTotal]
ellipseData[:,2] = rm
Moments0 = np.zeros((maxOffset+1,6))

for j in range(maxOffset + 1):
    Moments0[j,:] = getMoments(ellipseData[:(NxL + 2*j)**2,:], xG)


np.random.seed(10)
ns = 10
delta = 20.0
Moments = np.zeros((ns,maxOffset+1,6))
Moments_rel = np.zeros((ns,maxOffset+1,6))
ellipseData_accepted = np.zeros((ns,(NxL + 2*maxOffset)**2,5))

mean = np.loadtxt("mean.txt")
std = np.loadtxt("std.txt")

count = 0
# just to generate the distribution
while count < ns:
    ellipseData, permTotal, permBox = geni.circularRegular2Regions(r0, r1, NxL, NyL, Lx, Ly, maxOffset, ordered = False)
    ellipseData = ellipseData[permTotal]
    
    # reescaling (but taking the outer as it is)
    for i in range(offSetConstant+1): 
        ni =  (NxL + 2*(i-1))*(NxL + 2*(i-1))
        nout = (NxL + 2*i)*(NxL + 2*i) 
        alphaFrac = np.sqrt(((nout-ni)*H**2)*Vfrac/(np.pi*np.sum(ellipseData[ni:nout,2]**2)))
        ellipseData[ni:nout,2] = alphaFrac*ellipseData[ni:nout,2]
        
    ellipseData[(NxL + 2*offSetConstant)**2:,2] = rm
        
    MomentsTest = np.zeros((maxOffset+1,6))
    
    for j in range(maxOffset+1):
        MomentsTest[j,:] = getMoments(ellipseData[:(NxL + 2*j)**2,:],xG)

    if( np.max(np.abs((MomentsTest[1:,:] - Moments0[1:,:])/std[1:,:]).flatten()) < delta ):
        print("snapshot accepted")
        Moments[count,:,:] = copy.deepcopy(MomentsTest)
        ellipseData_accepted[count,:,:] = copy.deepcopy(ellipseData)
        Moments_rel[count,:,:] = (MomentsTest - mean)/std
        count = count + 1
    else:
        # print( np.abs((MomentsTest - Moments0)/std).flatten())E
        print("snapshot not accepted")

    
    
# mean = np.mean(Moments,axis = 0)
# std = np.mean(Moments,axis = 0)
# np.savetxt("mean.txt",mean)
# np.savetxt("std.txt",std)

plt.scatter(ellipseData_accepted[0,:36,0],ellipseData_accepted[0,:16,1],s=50000*ellipseData_accepted[0,:66,2])


plt.figure(1)
for i in range(6):
    plt.subplot('23' + str(i+1))
    plt.hist(Moments_rel[:,0,i])
    
plt.figure(2)
for i in range(6):
    plt.subplot('23' + str(i+1))
    plt.hist(Moments_rel[:,1,i])
    
plt.figure(3)
for i in range(6):
    plt.subplot('23' + str(i+1))
    plt.hist(Moments_rel[:,2,i])


    # ellipseData[permBox[4:],2] = outerRadius




