import numpy as np
import matplotlib.pyplot as plt
from subroutines import coll_der_mat

N = 64
x = np.linspace(0, 2*np.pi, N+1)[:-1]

D = coll_der_mat.fourier_coll_der_mat(N)
print(np.linalg.matrix_rank(D))

pf = D@D - np.cos(x)*D
u = np.dot(np.linalg.inv(pf), -np.sin(x)*np.exp(np.sin(x)))

plt.plot(x, np.real(u)/1e13, 'o-', color='salmon')
plt.xlabel('$x$')
plt.ylabel('$u$')
plt.show()
