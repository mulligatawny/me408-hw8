import numpy as np
import matplotlib.pyplot as plt
from transforms import cheby

def combine_coeffs(O, E):
    S = np.zeros(O.shape[0] + E.shape[0])
    S[::2] = E
    S[1::2] = O
    return S

nu = 0.5
dt = 0.001
N = 16
tc = np.linspace(0, np.pi, N+2)
x = np.cos(tc)
u0 = 1 - x**8
alpha = 2/(nu*dt)
gamma = np.ones(N+3)
gamma[-1] = 0
gamma[-2] = 0
gamma[-3] = 0
c = np.ones(N+1)
c[0] = 2
c[-1] = 2


Ae = np.zeros((int(N/2+1), int(N/2+1)), dtype='float')
ie = np.linspace(0, N, int(N/2+1)).astype(int)
je = ie[1:]
ke = ie[:-1]

# fill upper diagonal
np.fill_diagonal(Ae[:,1:], -1 - alpha*gamma[je]/(2*(je**2-1)))
# fill second upper diagonal
np.fill_diagonal(Ae[:,2:], alpha*gamma[je]/(4*je*(je+1)))
# fill leading diagonal
np.fill_diagonal(Ae, c[ke]*alpha/(4*je*(je-1)))
# fill bottom row with BC
Ae[-1,:] = np.ones(int(N/2+1))


Ao = np.zeros((int(N/2), int(N/2)), dtype='float')
io = np.linspace(1, N-1, int(N/2), dtype='float').astype(int)
jo = io[1:]
ko = io[:-1]

# fill leading diagonal
np.fill_diagonal(Ao, c[ko]*alpha/(4*jo*(jo-1)))
# fill upper diagonal
np.fill_diagonal(Ao[:,1:], -1 - alpha*gamma[jo]/(2*(jo**2-1)))
# fill second upper diagonal
np.fill_diagonal(Ao[:,2:], alpha*gamma[jo]/(4*jo*(jo+1)))
# fill BC row
Ao[-1,:] = np.ones(int(N/2))


u = cheby.cheby(u0)
tf = 0.5
t = 0.0

while t < 0.0013:
    ############### EVEN-NUMBERED N ####################################
    # RHS
    pr1e = alpha*u
    uie = cheby.icheby(u)
    pr2e = cheby.cheby(cheby.cheby_der(cheby.cheby_der(uie)))
    qe = -pr1e -pr2e
    qe = np.append(qe,0)
    qe = np.append(qe,0)
    be = np.zeros(int(N/2)+1)
    be[:-1] = be[:-1] + gamma[je]*qe[je]/(2*(je**2-1))
    be[:-1] = be[:-1] - c[ke]*qe[ke]/(4*je*(je-1))
    be[:-1] = be[:-1] - gamma[je+2]*qe[je+2]/(4*je*(je+1))
    be[-1] = 0.0
    print(be)
    phi_e = np.linalg.solve(Ae, be)

    ################ ODD-NUMBERED N ####################################
    # RHS
    pr1o = alpha*u
    uio = cheby.icheby(u)
    pr2o = cheby.cheby(cheby.cheby_der(cheby.cheby_der(uio)))
    qo = -pr1o -pr2o
    qo = np.append(qo,0)
    qo = np.append(qo,0)
    bo = np.zeros(int(N/2))
    bo[:-1] = bo[:-1] + gamma[jo]*qo[jo]/(2*(jo**2-1))
    bo[:-1] = bo[:-1] - c[ko]*qo[ko]/(4*jo*(jo-1))
    bo[:-1] = bo[:-1] - gamma[jo+2]*qo[jo+2]/(4*jo*(jo+1))
    bo[-1] = 0.0

    phi_o = np.linalg.solve(Ao, bo)

    un = combine_coeffs(phi_o, phi_e)
    u = un
    t = t + dt

ufinal = cheby.icheby(u)

plt.plot(np.real(ufinal))
#plt.show()
