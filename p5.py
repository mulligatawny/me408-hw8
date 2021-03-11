import numpy as np
import matplotlib.pyplot as plt

N = 32
x = np.linspace(0, 2*np.pi, N+1)[:-1]

D = np.zeros((N, N))

for l in range(N):
    for j in range(N):
        if l != j:
            D[l,j] = 0.5*(-1)**(l-j)/np.tan(np.pi*(l-j)/N)
        else:
            D[l,j] = 0.0

pf = D@D - np.cos(x)*D
u = np.dot(np.linalg.inv(pf), -np.sin(x)*np.exp(np.sin(x)))
plt.plot(x, np.real(u)/1e13, 'o-', color='salmon')
plt.xlabel('$x$')
plt.ylabel('$u$')
plt.show()
