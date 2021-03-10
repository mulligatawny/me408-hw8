import numpy as np

def sge(n, d0, dp1, dp2, row_n, rhs):
    """
    %--------------------------------------
    %  Performs a tridiagonal plus row
    %  Gaussian Elimination.
    %--------------------------------------
    % number of points
    % main diagonal
    % diagonal + 1
    % diagonal + 2
    % nth (bottom) row on input
    % right hand side on input
    % solution on output
    """
    n = np.int32(n)
    d0 = np.single(d0)
    dp1 = np.single(dp1)
    dp2 = np.single(dp2)
    row_n = np.single(row_n)
    rhs = np.single(rhs)
    soln = np.single(np.zeros((n,1)))

    tol = np.single(1.0e-7)
    i = np.int32(0)


    # forward elimination
    for i in range(0,n-1):
    # all operations are done to the bottom row
        if(abs(d0[i]) > tol):
            row_n[i+1] = row_n[i+1] - row_n[i]*dp1[i]/d0[i]

            if(i < n-2) :
                row_n[i+2] = row_n[i+2] - row_n[i]*dp2[i]/d0[i]  
            
            rhs[n-1] = rhs[n-1] - row_n[i]*rhs[i]/d0[i]
        else:
            print('SGE Error:  Matrix is Singular')
            return
        

    # backwards substitution
    if (abs(row_n[n-1]) > tol):
        rhs[n-1] = rhs[n-1]/row_n[n-1]
    else:
        print('SGE Error:  Matrix is Singular')
        return
    
    rhs[n-2] = (rhs[n-2] - dp1[n-2]*rhs[n-1])/d0[n-2]

    for i in range(n-3,-1,-1):
        rhs[i] =(rhs[i] - dp1[i]*rhs[i+1] - dp2[i]*rhs[i+2])/d0[i]
    
    soln=rhs
    # subroutine sge
    return soln