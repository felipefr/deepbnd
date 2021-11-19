import os, sys
sys.path.insert(0, '/home/felipefr/EPFL/newDLPDEs/MLELAS/utils')

import numpy as np
import meshUtils as meut
from dolfin import *
from mpi4py import MPI

folder = './'

solution_DNS = folder + 'barMacro_DNS_interp.xdmf' 

meshDNSfile =  folder + 'meshBarMacro_Multiscale_96.xdmf'
meshMultiscaleFile = 'meshBarMacro_Multiscale.xdmf'

meshDNS = Mesh()
# meshDNS = meut.EnrichedMesh(meshDNSfile)
with XDMFFile(meshDNSfile) as infile:
    infile.read(meshDNS)

print("mesh DNS loaded")

meshMultiscale = Mesh()
with XDMFFile(meshMultiscaleFile) as infile:
    infile.read(meshMultiscale)
        

Uh_DNS = VectorFunctionSpace(meshDNS, "CG", 2)
print(Uh_DNS.dim())
print("space DNS created")
Uh_mult = VectorFunctionSpace(meshMultiscale, "CG", 2)
print(Uh_mult.dim())

uh_DNS = Function(Uh_DNS)
with XDMFFile(solution_DNS) as infile:
    infile.read_checkpoint(uh_DNS, 'u', 0)

uh_DNS_interp = interpolate(uh_DNS,Uh_mult)

with XDMFFile("barMacro_DNS.xdmf") as file:
    file.write_checkpoint(uh_DNS_interp,'u',0)
    