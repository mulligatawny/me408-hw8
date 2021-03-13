# 4/4
import numpy as np
import matplotlib.pyplot as plt
from transforms import cheby

def cheby_der(f):
    N = len(f)-1
    # compute chebyshev transform
    Fk = cheby.cheby(f)
    k = np.arange(0, N+1)
    # assemble bi-diagonal matrix
    A = np.zeros((N+1, N+1))
    np.fill_diagonal(A[1:], 1)
    np.fill_diagonal(A[:,1:], -1)
    A[0,:] = 0
    A[1,0] = 2
    nA = A[1:,:-1]
    # assmble RHS
    b = np.zeros(N+1)
    b = 2*k*Fk
    bn = b[1:]
    # solve bi-diagonal system
    phi = np.linalg.solve(nA, bn)
    # set last coefficient to 0
    phi = np.append(phi, 0.0)
    # inverse transform
    return cheby.icheby(phi)

N = 32
t = np.arange(0, N+1)*np.pi/N
x = np.cos(t)
f = x**4
#fp = cheby_der(f)
fp = cheby.cheby_der(f)
#plt.plot(x, fp, '-o', label='N = {}'.format(N))
k = np.arange(0, N+1)
plt.plot(k, fp)

# exact derivative
#x = np.linspace(-1, 1, 128)
#plt.plot(x, dfdx, '.', label='exact')
plt.xlabel('$x$')
plt.ylabel('$df$/$dx$')
plt.legend()
plt.show()

