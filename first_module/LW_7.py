import numpy as np
from math import *
import matplotlib.pyplot as plt

x0 = float(input('Начальное отклонение маятника '))
v0 = float(input('Начальная скорость маятника '))
f0 = float(input('Частота колебаний маятника '))
T = float(input('Временной интервал '))
n = int(input('Число временных отсчетов '))

t = np.linspace(0, T, n)
ax = [0] * n
av = [0] * n

av[0] = v0
ax[0] = x0
dt = T / (n - 1)
for i in range(1, n):
    av[i] = av[i - 1] - (2.0 * pi * f0) ** 2 * ax[i - 1] * dt
    ax[i] = ax[i - 1] + av[i] * dt

spec = np.fft.fft(ax)
freq = np.fft.fftfreq(n, T / n)

plt.plot(t, ax)
plt.axis(xmin=0, xmax=T)
plt.show()

plt.plot(t, av)
plt.axis(xmin=0, xmax=T)
plt.show()

plt.plot(freq[0:n // 2], (np.hypot(spec.real, spec.imag) / n * 2.0)[0:n // 2], '-.', color='tab:orange', label='БПФ')
plt.axis(xmin=0, xmax=2 * f0)
plt.show()