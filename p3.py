###############################################################################
# 1D advection solver using second-order finite differences and Runge-Kutta   #
# IV time integration                                                         #
###############################################################################

import numpy as np
import matplotlib.pyplot as plt

N = 512
x = np.linspace(0, 2*np.pi, N)
dx = 2*np.pi/(N-1)
u0 = np.exp(-100*(x-2)**2)

def fun(t, u):
    
    c = 1/10 + 1/2*(np.sin(x[i]-2))**2
    if i == 0:
        f = -c*(u[1] - u[-1])/(2*dx)
    if i == N-1:
        f = -c*(u[0] - u[-2])/(2*dx)
    else:
        f =  -c*(u[i+1] - u[i-1])/(2*dx)
    return f
    
un = np.zeros_like(x, dtype='float')

dt = 0.001
t = 0.0
tf = 16
u = u0

while t < tf:
    for i in range(N):
        k1 = dt*fun(t, u)
        k2 = dt*fun(t+dt/2, u+k1/2)
        k3 = dt*fun(t+dt/2, u+k2/2)
        k4 = dt*fun(t+dt, u+k3)
        un[i] = u[i] + k1/6 + k2/3 + k3/3 + k4/6

    u = un
    t = t + dt

plt.plot(x, u0, 'k',label='I.C')
plt.plot(x, u, 'ro-', label='F.D.')
plt.title('N = {}'.format(N))
plt.xlabel('$x$')
plt.ylabel('$u$')
plt.legend()
plt.show()

np.save('fd.npy', u)
