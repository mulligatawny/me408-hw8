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
