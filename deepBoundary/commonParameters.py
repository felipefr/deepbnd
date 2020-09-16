# this file is just to set standards parameters that will be used elsewhere in this folder

def getCommonInclusionsExample():
    offset = 2
    Lx = Ly = 1.0
    ifPeriodic = False 
    NxL = NyL = 2
    NL = NxL*NyL
    x0L = y0L = offset*Lx/(NxL+2*offset)
    LxL = LyL = offset*Lx/(NxL+2*offset)
    r0 = 0.2*LxL/NxL
    r1 = 0.4*LxL/NxL
    times = 1
    lcar = 0.1*LxL/NxL
    NpLx = int(Lx/lcar) + 1
    NpLxL = int(LxL/lcar) + 1
    Vfrac = 0.282743
    
    contrast = 10.0
    E2 = 1.0
    E1 = contrast*E2 # inclusions
    nu1 = 0.3
    nu2 = 0.3 
    
    
    d = {'offset': offset, 
            'Lx' : Lx,
            'Ly' : Ly, 
            'ifPeriodic': ifPeriodic , 
            'NxL': NxL,
            'NyL': NyL,
            'NL': NxL*NyL,
            'x0L' : x0L,
            'y0L' : y0L,
            'LxL' : LxL, 
            'LyL' : LyL,
            'r0' : r0,
            'r1' : r1,
            'times' : 1,
            'lcar' : lcar, 
            'NpLx' : NpLx,
            'NpLx' : NpLxL,
            'Vfrac' : Vfrac,
            'contrast' : contrast,
            'E2' : E1,
            'E1' : E2, 
            'nu1' : nu1,
            'nu2' : nu2 }
    
    return d 