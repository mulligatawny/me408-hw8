import numpy as np

def mge(n, d0, dp1, dp2, row_n, rhs):
    """
    %--------------------------------------
    %  Performs a tridiagonal plus row
    %  Gaussian Elimination that shifts
    %  a_0 to a_16 and a_n to a_n-2 for all
    %  other n.
    %--------------------------------------
    % number of points
    % main diagonal on input
    % diagonal + 1 on input
    % diagonal + 2 on input
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

    col = np.single(np.zeros((n-1)))
    i = np.int32(0)
    temp = np.single(0.0)
    tol = np.single(1.0e-7)

    # rearrange data
    temp = row_n[0]
    for i in range(0, n-1):
        row_n[i] = row_n[i+1]

    row_n[n-1] = temp

    col[0] = d0[0]
    # forward elimination
    for i in range(0, n-2):
        if (abs(dp1[i]) > tol):
    # eliminate next row
            if(i <= n-3):
                dp1[i+1] = dp1[i+1] - d0[i+1]*dp2[i]/dp1[i]
                col[i+1] =          - d0[i+1]*col[i]/dp1[i]
                rhs[i+1] = rhs[i+1] - d0[i+1]*rhs[i]/dp1[i]
            
    # eliminate bottom row
            row_n[i+1] = row_n[i+1] - row_n[i]*dp2[i]/dp1[i]
            row_n[n-1] = row_n[n-1] - row_n[i]*col[i]/dp1[i]
            rhs[n-1] = rhs[n-1] - row_n[i]*rhs[i]/dp1[i]
        else:
            print('MGE Error:  Matrix is Singular')
            return
        
    # do row n-1, note dp2(n-1) = 0
    row_n[n-1] = row_n[n-1] - row_n[n-2]*col[n-2]/dp1[n-2]
    rhs[n-1] = rhs[n-1] - row_n[n-2]*rhs[n-2]/dp1[n-2]

    # backwards substitution
    if (abs(row_n[n-1]) > tol):
        rhs[n-1] = rhs[n-1]/row_n[n-1]
    else:
        print('MGE Error:  Matrix is Singular')
        return
    
    rhs[n-2] = (rhs[n-2] - col[n-2]*rhs[n-1])/dp1[n-2]
    for i in range(n-3,-1,-1):
        rhs[i] =(rhs[i] - dp2[i]*rhs[i+1] - col[i]*rhs[n-1])/dp1[i]

    soln = np.hstack([rhs[-1], rhs[0:-1]])
    return soln