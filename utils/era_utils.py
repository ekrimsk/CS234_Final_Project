import numpy as np

def binvec(control, n):  
    # assert input of type int   
    assert isinstance(n, int)
    out = np.zeros(n)
    for ii in range(n):
        out[ii] = control & 1                
        control = control >> 1 

    return out
