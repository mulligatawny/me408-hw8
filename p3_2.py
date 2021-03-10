###############################################################################
# 1D advection solver using collocation method with Fourier expansions and    #
# Runge-Kutta IV time integration                                             #
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

plot = 1 # toggle between 2D/3D plots

N = 512
nu = 0.5
x = np.linspace(0, 2*np.pi, N)
u0 = np.exp(-100*(x-2)**2)
c = 1/10 + 1/2*(np.sin(x-2))**2

def fun(t, u):
    uk = np.fft.fftshift(np.fft.fft(u))
    k = np.arange(-N/2, N/2)
    ukp = 1j*k*uk
    up = np.fft.ifft(np.fft.ifftshift(ukp))
    return -c*up

dt = 0.01
t = 0.0
tf = 16
u = u0

# variables to save off 3D plot slices
inc = 0
inc2 = 0
phi = np.zeros((N, 100))

while t < tf:
    k1 = dt*fun(t, u)
    k2 = dt*fun(t+dt/2, u+k1/2)
    k3 = dt*fun(t+dt/2, u+k2/2)
    k4 = dt*fun(t+dt, u+k3)
    un = u + k1/6 + k2/3 + k3/3 + k4/6
    
    inc = inc+1
    # save off every 16th slice
    if inc%16 == 0:
        phi[:,inc2] = np.real(un)
        inc2 = inc2 + 1
        inc = 0

    u = un
    t = t + dt

u_fd = np.load('fd.npy') # load finite difference solution

if not plot:
    plt.plot(x, np.real(un), 'ro-', label='Colloc.')
    plt.plot(x, np.real(u_fd), 'k-', label='F.D')
    plt.xlabel('$x$')
    plt.ylabel('$u(t = 16)$')
    plt.title('N = {}'.format(N))
    plt.legend()

t = np.linspace(0, tf, 100)

# plot 3D surface
if plot:
    T, X = np.meshgrid(t, x)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(T, X, phi, cmap=cm.inferno,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('$T$')
    ax.set_ylabel('$x$')
plt.show()
