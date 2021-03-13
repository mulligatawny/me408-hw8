import numpy as np
import matplotlib.pyplot as plt
from ge import mge
from ge import sge

N = 6
alpha = 0.5
gamma = np.ones(N+1)
gamma[-1] = 0
gamma[-2] = 0
c = np.ones(N+1)
c[0] = 2
c[-1] = 2

# even-numbered n
A = np.zeros((int(N/2+1), int(N/2+1)), dtype='float')
i = np.linspace(0, N, int(N/2+1)).astype(int)
j = i[1:]
k = i[:-1]

# fill upper diagonal
np.fill_diagonal(A[:,1:], -1 - alpha*gamma[j]/(2*(j**2-1)))
# fill second upper diagonal
np.fill_diagonal(A[:,2:], alpha*gamma[j]/(4*j*(j+1)))
# fill leading diagonal
np.fill_diagonal(A, c[k]*alpha/(4*j*(j-1)))
# fill bottom row with BC
A[-1,:] = 2*i**2
# RHS
b = np.zeros(int(N/2)+1)
b[-1] = 2*0.75

print(A)
# solve using pivoting, mge and sge
phi_e_exact = np.linalg.solve(A, b)
phi_e_mge = mge.mge(int(N/2+1), np.diagonal(A), np.diagonal(A,1), \
                    np.diagonal(A,2), A[-1,:], b)
phi_e_sge = sge.sge(int(N/2+1), np.diagonal(A), np.diagonal(A,1), \
                    np.diagonal(A,2), A[-1,:], b)

np.save('./data/p1/phi_e_exact.npy', phi_e_exact)
np.save('./data/p1/phi_e_mge.npy', phi_e_mge)
np.save('./data/p1/phi_e_sge.npy', phi_e_sge)
