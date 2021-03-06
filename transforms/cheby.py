#=============================================================================#
# Discrete Chebyshev Transform (Forward and Reverse) from Moin P. pp. 190     #
#=============================================================================#

import numpy as np

def cheby(f):
    """
    Computes the 1D discrete Chebyshev transform of f
    Parameters:
        f  (array_like) : function
    Returns:
        Fk (array_like) : Chebyshev coefficients

    """
    N = int(len(f))-1
    Fk = np.zeros_like(f, dtype='float')
    t = np.arange(0, N+1)*np.pi/N # uniform grid in theta
    
    for k in range(N+1):
        cs = np.cos(k*t)
        cs[0] = cs[0]/2
        cs[-1] = cs[-1]/2
        Fk[k] = np.dot(f,cs)/N*2
    Fk[0] = Fk[0]/2
    Fk[-1] = Fk[-1]/2
    return Fk

def icheby(Fk):
    """
    Computes the 1D discrete inverse Chebyshev transform of f
    Parameters:
        Fk (array_like) : Chebyshev coefficients
    Returns:
        fc (array_like) : reconstructed function 

    """
    N = int(len(Fk))-1
    fc = np.zeros_like(Fk, dtype='float')
    t = np.arange(0, N+1)*np.pi/N # uniform grid in theta

    for k in range(N+1):
        fc = fc + Fk[k]*np.cos(k*t)
    return fc
