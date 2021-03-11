import numpy as np
import matplotlib.pyplot as plt

def compute_error(N):
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

    print('Beginning time integration with N = {}'.format(N))
    if N > 1000:
        dt = 0.002
    else:
        dt = 0.01
    t = 0.0
    tf = 10*np.sqrt(2/3)*np.pi
    u = u0

    while t < tf:
        k1 = dt*fun(t, u)
        k2 = dt*fun(t+dt/2, u+k1/2)
        k3 = dt*fun(t+dt/2, u+k2/2)
        k4 = dt*fun(t+dt, u+k3)
        un = u + k1/6 + k2/3 + k3/3 + k4/6

        u = un
        t = t + dt
    # one last step to exactly reach tf
    t = t - dt
    dt = tf - t
    k1 = dt*fun(t, u)
    k2 = dt*fun(t+dt/2, u+k1/2)
    k3 = dt*fun(t+dt/2, u+k2/2)
    k4 = dt*fun(t+dt, u+k3)
    un = u + k1/6 + k2/3 + k3/3 + k4/6
    u  = un
    t = t + dt
    print('Completed. Time is {}'.format(t))
    return np.max(np.abs(u0 - u))

N = np.array([32, 64, 128, 256, 512, 1024, 2048])
e = np.zeros(len(N))

for i in range(len(N)):
    e[i] = compute_error(N[i])

plt.loglog(N, e, 'ro-', label='colloc')
plt.loglog(N, 1/N*4.1, 'g-', label='slope-1')
plt.xlabel('$N$')
plt.ylabel('Error')
plt.grid(which='both')
plt.legend()
plt.show()
