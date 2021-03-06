import numpy as np
import matplotlib.pyplot as plt
from transforms import cheby

N = 16
alpha = 0.5
gamma = np.ones(N+1)
gamma[-1] = 0
gamma[-2] = 0
c = np.ones(N+1)
c[0] = 2
c[-1] = 2

i = np.linspace(1, N-1, int(N/2), dtype='float').astype(int)
j = i[1:]
k = i[:-1]


A = np.zeros((int(N/2), int(N/2)), dtype='float')

# odd-numbered n
np.fill_diagonal(A, c[k]*alpha/(4*j*(j-1)))
np.fill_diagonal(A[:,1:], -1 - alpha*gamma[j]/(2*(j**2-1)))
np.fill_diagonal(A[:,2:], alpha*gamma[j]/(4*j*(j+1)))
A[-1,:] = 2*i**2

b = np.zeros(int(N/2))
b = 2/i**2
b[-1] = 0

phi_o = np.linalg.solve(A, b)

phi_e = np.load('even.npy')

phi = np.zeros(phi_e.shape[0] + phi_o.shape[0], dtype='float')
phi[::2] = phi_e
phi[1::2] = phi_o

fc = cheby.icheby(phi)
plt.plot(np.real(fc))
#plt.plot(abs(phi), 'o-',color='orangered')
plt.show()
