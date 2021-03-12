import numpy as np
import matplotlib.pyplot as plt

N = 128
#t = np.linspace(0, np.pi, N+1)
#x = np.flip(np.cos(t))
x = np.linspace(-1, 1, N+1)
u0 = np.zeros(N+1)

t = 0
tf = 0.9
dt = 0.001
dx = 2/N
u = u0
un = np.zeros_like(x)
        
while t < tf: 
    
    if t < 0.5:
        u[0] = np.sin(2*np.pi*t)
    else:
        u[0] = 0

    A = np.zeros((N+1, N+1))
    np.fill_diagonal(A, 1)
    A[0,0] = (1 - dt/dx)
    A[0,1] = A[0,1]*2
    np.fill_diagonal(A[1:], -dt/(2*dx))
    # upper
    np.fill_diagonal(A[:,1:], dt/(2*dx))
    A[-1,-1] = 1 + dt/dx
    A[-1,-2] = A[-1,-2]*2

    b = np.zeros_like(x)
    for j in range(N+1):
        if j==0:
            b[0] = u[0] -dt/dx*(u[j+1] - u[j])
        elif j==N:
            b[N] = u[N] -dt/dx*(u[j] - u[j-1])
        else:
            b[j] = u[j] - dt/(2*dx)*(u[j+1] - u[j-1])

    un = np.linalg.solve(A, b)
    u = un
    t = t + dt

# exact solution
ue = np.sin(2*np.pi*(tf - x/2 - 1/2))
ue[x>2*tf-1] = 0
ue[x<2*tf-2] = 0

plt.plot(x, ue)
plt.plot(x, u0)
plt.plot(x, un)
plt.show()
