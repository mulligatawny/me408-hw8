import numpy as np
import matplotlib.pyplot as plt
from ge import mge
from ge import sge

N = 16
alpha = 0.5
gamma = np.ones(N+2)
gamma[-1] = 0
gamma[-2] = 0
gamma[-3] = 0
c = np.ones(N+1)
c[0] = 2
c[-1] = 2

# odd-numbered n
A = np.zeros((int(N/2), int(N/2)), dtype='float')
i = np.linspace(1, N-1, int(N/2), dtype='float').astype(int)
ip = i[1:]
im = i[:-1]

# fill leading diagonal
np.fill_diagonal(A, c[im]*alpha/(4*ip*(ip-1)))
# fill upper diagonal
np.fill_diagonal(A[:,1:], -1 - alpha*gamma[ip]/(2*(ip**2-1)))
# fill second upper diagonal
np.fill_diagonal(A[:,2:], alpha*gamma[ip]/(4*ip*(ip+1)))
# fill BC row
A[-1,:] = 2*i**2
# RHS
b = np.zeros(int(N/2))
p = np.arange(N+2)
q = 2/p**2
q[0] = 0
b[:-1] = b[:-1] + gamma[ip]*q[ip]/(2*(ip**2-1))
b[:-1] = b[:-1] - c[im]*q[im]/(4*ip*(ip-1))
b[:-1] = b[:-1] - gamma[ip+2]*q[ip+2]/(4*ip*(ip+1))
b[-1] = 0.0

# solve using pivoting, mge and sge
phi_o_exact = np.linalg.solve(A, b)
phi_o_mge = mge.mge(int(N/2), np.diagonal(A), np.diagonal(A,1), \
                    np.diagonal(A,2), A[-1,:], b)
phi_o_sge = sge.sge(int(N/2), np.diagonal(A), np.diagonal(A,1), \
                    np.diagonal(A,2), A[-1,:], b)

# load even n data
phi_e_exact = np.load('./data/p1/phi_e_exact.npy')
phi_e_mge = np.load('./data/p1/phi_e_mge.npy')
phi_e_sge = np.load('./data/p1/phi_e_sge.npy')

def combine_coeffs(O, E):
    S = np.zeros(O.shape[0] + E.shape[0])
    S[::2] = E
    S[1::2] = O
    return S

phi_exact = combine_coeffs(phi_o_exact, phi_e_exact)
phi_mge = combine_coeffs(phi_o_mge, phi_e_mge)
phi_sge = combine_coeffs(phi_o_sge, phi_o_sge)

plt.plot(abs(phi_exact), 'o-',color='orangered', label='pivoting')
plt.plot(abs(phi_mge), 'x-',color='darkblue', label='MGE')
plt.plot(abs(phi_sge), '-',color='green', label='SGE')
plt.ylim([-0.1, 2])
plt.xlabel('$n$')
plt.ylabel('$a_n$')
plt.legend()
plt.show()
