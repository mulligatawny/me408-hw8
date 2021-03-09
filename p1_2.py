import numpy as np
import matplotlib.pyplot as plt
from transforms import cheby

np.set_printoptions(precision=5)
N = 16
alpha = 0.5
gamma = np.ones(N+2)
gamma[-1] = 0
gamma[-2] = 0
gamma[-3] = 0
c = np.ones(N+1)
c[0] = 2
c[-1] = 2


i = np.linspace(1, N-1, int(N/2), dtype='float').astype(int)
ip = i[1:]
im = i[:-1]

A = np.zeros((int(N/2), int(N/2)), dtype='float')

# odd-numbered n
np.fill_diagonal(A, c[im]*alpha/(4*ip*(ip-1)))
np.fill_diagonal(A[:,1:], -1 - alpha*gamma[ip]/(2*(ip**2-1)))
np.fill_diagonal(A[:,2:], alpha*gamma[ip]/(4*ip*(ip+1)))
A[-1,:] = 2*i**2

b = np.zeros(int(N/2))

p = np.arange(N+2)
q = 2/p**2
q[0] = 0

b[:-1] = b[:-1] + gamma[ip]*q[ip]/(2*(ip**2-1))
b[:-1] = b[:-1] - c[im]*q[im]/(4*ip*(ip-1))
b[:-1] = b[:-1] - gamma[ip+2]*q[ip+2]/(4*ip*(ip+1))

b[-1] = 0.0

phi_o = np.linalg.solve(A, b)
#print(phi_o)
phi_e = np.load('even.npy')
#print(A)
phi = np.zeros(phi_e.shape[0] + phi_o.shape[0], dtype='float')
phi[::2] = phi_e
phi[1::2] = phi_o

plt.plot(abs(phi), 'o-',color='orangered')
plt.show()
