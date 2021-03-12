import numpy as np
import matplotlib.pyplot as plt
from subroutines import coll_der_mat
from sklearn.metrics import mean_squared_error

def spectral_sol(N):
    t = np.linspace(0, np.pi, N+1)
    x = np.cos(t)
    u0 = np.zeros(N+1)
    D = coll_der_mat.cheby_coll_der_mat(N, x)
    t = 0.0
    dt = 0.00001
    tf = 0.9
    u = u0
    un = np.zeros(N+1)

    while t < tf:
        pf1 = np.linalg.inv(np.eye(N+1) + dt*D)
        pf2 = np.eye(N+1) - dt*D
        pf = pf1 @ pf2
        un = np.dot(pf, u)
        # BC
        if t < 0.5:
            un[-1] = np.sin(2*np.pi*t)
        else:
            un[-1] = 0
        u = un
        t = t + dt

    ue = np.sin(2*np.pi*(tf - x/2 - 1/2))
    ue[x>2*tf-1] = 0
    ue[x<2*tf-2] = 0

    return mean_squared_error(u, ue)

def FD_sol(N):

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

    return mean_squared_error(u, ue)

N = np.array([8, 16, 32, 64, 128])
err = np.zeros(len(N))

for i in range(len(N)):
#    err[i] = spectral_sol(N[i])
    err[i] = FD_sol(N[i])
    
plt.loglog(N,err, 'o-', color='orangered', label='finite diff.')
plt.loglog(N, 2/N**2, 'k-', label='slope-2')
plt.xlabel('$N$')
plt.ylabel('RMS error')
plt.grid(which='both')
plt.legend()
plt.show()
