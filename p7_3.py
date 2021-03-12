import numpy as np
import matplotlib.pyplot as plt

N = 1024
#t = np.linspace(0, np.pi, N+1)
#x = np.flip(np.cos(t))
x = np.linspace(-1,1, N+1)
u0 = np.zeros(N+1)

t = 0
tf = 0.5
dt = 0.0005
dx = 2/N
u = u0
un = np.zeros(N+1)
        
def fun(t, u):
    if i == 0:
        f = -(u[1] - u[-2])/dx
    elif i == N:
        f = -(u[0] - u[N-2])/dx
    else:
        f = -(u[i+1] - u[i-1])/dx
    return f

while t < tf: 
    for i in range(N):
        k1 = dt*fun(t, u)
        k2 = dt*fun(t+dt/2, u+k1/2)
        k3 = dt*fun(t+dt/2, u+k2/2)
        k4 = dt*fun(t+dt, u+k3)
        un[i] = u[i] + k1/6 + k2/3 + k3/3 + k4/6
    if t < 0.5:
        un[0] = np.sin(2*np.pi*t)
    else:
        un[0] = 0
    u = un
    t = t + dt



# exact solution
ue = np.sin(2*np.pi*(tf - x/2 - 1/2))
ue[x>2*tf-1] = 0
ue[x<2*tf-2] = 0
plt.plot(x, u)
plt.plot(x, ue)
plt.show()
