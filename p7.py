import numpy as np
import matplotlib.pyplot as plt
from subroutines import coll_der_mat

N = 32
t = np.linspace(0, np.pi, N+1)
x = np.cos(t)
u0 = np.zeros(N+1)
D = coll_der_mat.cheby_coll_der_mat(N, x)
t = 0.0
dt = 0.001
tf = 0.9
u = u0
un = np.zeros(N+1)

while t < tf:
    pf1 = np.linalg.inv(np.eye(N+1) + dt*D)
    pf2 = np.eye(N+1) - dt*D
    pf = pf1 @ pf2
    un = np.dot(pf, u)
    if t < 0.5:
        un[-1] = np.sin(2*np.pi*t)
    else:
        un[-1] = 0
    u = un
    t = t + dt

ue = np.sin(2*np.pi*(tf - x/2 - 1/2))
ue[x>2*tf-1] = 0
ue[x<2*tf-2] = 0
plt.plot(x, ue)
plt.plot(x, u)
plt.show()
