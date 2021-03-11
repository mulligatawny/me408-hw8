import numpy as np
import matplotlib.pyplot as plt
from subroutines import coll_der_mat

N = 32
t = np.linspace(0, np.pi, N+1)
x = np.cos(t)
f = x**8 + x**5
fpe = 8*x**7 + 5*x**4 # exact derivative
fppe = 8*7*x**6 + 5*4*x**3

D = coll_der_mat.cheby_coll_der_mat(N, x)
fp = np.dot(D, f)
fpp = np.dot(D, fp)
fp_int = np.dot(np.linalg.inv(D), fp)

fig1 = plt.figure(1)
plt.plot(x, fp, 'Xk', label='cheby')
plt.plot(x, fpe, '-', color='mediumspringgreen', label='exact')
plt.xlabel('$x$')
plt.ylabel('$f\'$')
plt.legend()
plt.title('$f(x) = x^{8} + x^{5}$')
plt.show()

fig2 = plt.figure(2)
plt.plot(x, fpp, 'Xk', label='cheby')
plt.plot(x, fppe, '-', color='mediumspringgreen', label='exact')
plt.xlabel('$x$')
plt.ylabel('$f\'\'$')
plt.legend()
plt.title('$f(x) = x^{8} + x^{5}$')
plt.show()

fig3 = plt.figure(3)
plt.plot(x, fp_int, 'Xk', label='cheby')
plt.plot(x, f, '-', color='mediumspringgreen', label='exact')
plt.xlabel('$x$')
plt.ylabel('$f\'$')
plt.legend()
plt.title('f by integration of f\'')
plt.show()
