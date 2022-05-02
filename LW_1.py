import numpy as np
from math import *
import matplotlib.pyplot as plt

def signal(x):
    return 1.0 + sin(2.0*pi*x) + 2.0 * cos(4.0*pi*x) + 0.5 * cos(6.0*pi*x)

f = float(input('Опорная частота сигнала '))
T = float(input('Временной интервал '))
n = int(input('Число временных отсчетов '))

fp = open('data.csv', 'w', encoding='utf-8')

t = np.linspace(0, T, n)
u = [0] * n
fp.write('t; u\n')
for i in range(n):
    u[i] = signal(f*t[i])
    fp.write('{0: f}; {1: f}\n'.format(t[i], u[i]))
fp.close()

plt.plot(t, u)
#plt.axis(xmin=0, xmax=t[n-1])
#plt.plot(t, u, 'o', color='tab:orange')
#plt.vlines(t, 0, u, color='tab:orange', lw=1)
#plt.hlines(0, 0, t[n-1], color='tab:orange', lw=1)
plt.show()