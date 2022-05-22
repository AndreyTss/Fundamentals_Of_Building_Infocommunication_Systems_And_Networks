import numpy as np
from math import *
import matplotlib.pyplot as plt

def signal(x):
    return 1.0 + np.sin(2.0*pi*x) + 2.0 * np.cos(4.0*pi*x) + 0.5 * np.cos(6.0*pi*x)

#Дискретное преобразование Фурье
def dft(data):
    n = len(data)
    spectr = [0+0j] * n
    coef = 2.0 / n
    arg = pi * coef
    for j in range(n):
        spectr[j] = 0.0 + 0.0j
        for i in range(n):
            spectr[j] += data[i] * (cos(arg*i*j) - sin(arg*i*j)*1j)
    return np.array(spectr, dtype=complex)

f = float(input('Опорная частота сигнала '))
T = float(input('Временной интервал '))
n = int(input('Число временных отсчетов '))
print('\nПериод дискретизации: {0: f} сек\nЧастота дискретизации: {1: f} Гц'.format(T/n, n/T))
print('Максимальная частота в спектре сигнала: {0: f} Гц'.format(3*f))
print('Минимальный периоддискретизации: {0: f} сек.\n'.format(1/6/f))

t = np.linspace(0, T, n)
u = signal(f*t)

print('Расчет ДПФ...')
spec = dft(u)

freq = np.linspace(0, n/T, n)

#spec[0] = spec[0].real/2 + 1j*spec[0].imag

fig, ax = plt.subplots()
ax.plot(t, u)
ax.vlines(t, 0, u, color='tab:orange', lw=1)
ax.hlines(0, 0, t[n-1], color='tab:orange', lw=1)
ax.set_xlabel("$t$, сек.", fontsize=10)
ax.set_ylabel("$U(t)$, В", fontsize=10)
#ax.set_xlim(0, 1/f)
#plt.show()

fig, ax = plt.subplots()
ax.plot(freq, (np.hypot(spec.real, spec.imag)/n*2.0), color='tab:blue', label='ДПФ')
#ax.plot(freq[0:n//2], (np.hypot(spec.real, spec.imag)/n*2.0)[0:n//2], color='tab:blue', label='ДПФ')
ax.set_xlabel("$f$, Гц", fontsize=10)
ax.set_ylabel("$U(f)$, В", fontsize=10)
ax.legend(loc='best')
#ax.set_xlim(0, 3.5*f)
#plt.show()

fig, ax = plt.subplots()
ax.plot(freq, (np.arctan2(spec.imag, spec.real)), color='tab:orange', label='ДПФ')
#ax.plot(freq[0:n//2], (np.arctan2(spec.imag, spec.real))[0:n//2], color='tab:orange', label='ДПФ')
ax.set_xlabel("$f$, Гц", fontsize=10)
ax.set_ylabel("$\phi(f)$, рад.", fontsize=10)
ax.legend(loc='best')
#ax.set_xlim(0, 3.5*f)
plt.show()