###############################################################################
# Matrix Operators for Spectral Numerical Differentiation (Moin pp. 185-194)  #
###############################################################################

import numpy as np

def fourier_coll_der_mat(N):
    """
    Computes the matrix form of the Fourier Collocation Derivative.
    Parameters:
        N (int)     : number of collocation points
    Returns:
        D (2D array): Fourier collocation derivative matrix

    """
    D = np.zeros((N, N))

    for l in range(N):
        for j in range(N):
            if l != j:
                D[l,j] = 0.5*(-1)**(l-j)/np.tan(np.pi*(l-j)/N)
            else:
                D[l,j] = 0.0
    return D

def cheby_coll_der_mat(N, x):
    """
    Computes the matrix form of the Chebyshev Collocation Derivative
    Parameters:
        N (int)        : number of collocation points
        x (array_like) : Chebyshev grid points 
    Returns:
        D (2D array)   : Chebyshev collocation derivative matrix

    """
    c = np.ones(N+1)
    c[0] = 2
    c[-1] = 2
    D = np.zeros((N+1,N+1))

    for j in range(N+1):
        for k in range(N+1):
            if j==k and j==N:
                D[j][k] = -(2*N**2+1)/6
            elif j==k and j==0:
                D[j][k] = (2*N**2+1)/6
            elif j==k and j!=0 and j!=N:
                D[j][k] = -x[j]/(2*(1-x[j]**2))
            elif j!=k:
                D[j][k] = (c[j]*(-1)**(j+k))/(c[k]*(x[j]-x[k]))
    return D
