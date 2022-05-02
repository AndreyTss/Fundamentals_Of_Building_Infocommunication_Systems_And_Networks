import numpy as np
from math import *
import matplotlib.pyplot as plt
import time

def signal(x):
    return np.sin(2.0*pi*x) + 2.0 * np.cos(4.0*pi*x) + 0.5 * np.cos(6.0*pi*x)

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


#Обратное дискретное преобразование Фурье
def idft(data):
    n = len(data)
    signal = [0+0j] * n
    coef = 2.0 / n
    arg = pi * coef
    for j in range(n):
        signal[j] = 0.0 + 0.0j
        for i in range(n):
            signal[j] += data[i] * (cos(arg*i*(j+0)) + sin(arg*i*(j+0))*1j)
    return np.array(signal, dtype=complex)/n

use_only_fft = False #True - Используем только БПФ (ОБПФ)
ref_freq = False     #True - Отрицательные частоты для побочной полосы
full_scale = False   #True - Отображать графики в полном масштабе по оси абсцисс

f = float(input('Опорная частота сигнала '))
T = float(input('Временной интервал '))
n = int(input('Число временных отсчетов '))

print('\nМаксимальная частота в спектре сигнала: {0:f} усл. Гц'.format(3*f)) #определяется самим сигналом
print('Заданный шаг (частота) дискретизации: {0:f} усл. сек. ({1:f} усл. Гц)'.format(T/n, n/T))
print('Максимальный шаг дискретизации по В.А. Котельникову: {0:f} усл. сек.\n'.format(1/6/f))

t = np.linspace(0, T, n)
u = signal(f*t)

if not use_only_fft:
    print('Расчет ДПФ...')
    t0 = time.time()
    spec1 = dft(u)
    tdft = time.time()-t0
    print('Время расчета спектра сигнала по алгоритму ДПФ: {0:f} сек'.format(tdft))

print('Расчет БПФ...')
t0 = time.time()
spec2 = np.fft.fft(u)
tfft = time.time()-t0
print('Время расчета спектра сигнала по алгоритму БПФ: {0:f} сек'.format(tfft))

freq = np.fft.fftfreq(n, T/n) if ref_freq else np.linspace(0, n/T, n)

if not use_only_fft:
    print('Расчет ОДПФ...')
    sig_dft = idft(spec1)
print('Расчет ОБПФ...')
sig_fft = np.fft.ifft(spec2)

fig, ax = plt.subplots()
ax.plot(t, u, label='Непрерывный')
ax.vlines(t, 0, u, color='tab:orange', lw=1, label='Дискретный')
ax.hlines(0, 0, t[n-1], color='tab:orange', lw=1)
ax.set_xlabel("$t$, отн. ед.", fontsize=10)
ax.set_ylabel("$U(t)$, В", fontsize=10)
ax.legend(loc='best')
if not full_scale:
    ax.set_xlim(0, 1/f)

fig, ax = plt.subplots()
if not use_only_fft:
    ax.plot(freq[0:n//2], (np.hypot(spec1.real, spec1.imag)/n*2.0)[0:n//2], color='tab:blue', label='ДПФ')
    ax.plot(freq[n//2:n], (np.hypot(spec1.real, spec1.imag)/n*2.0)[n//2:n], color='tab:blue')
ax.plot(freq[0:n//2], (np.hypot(spec2.real, spec2.imag)/n*2.0)[0:n//2], '-.', color='tab:orange', label='БПФ')
ax.plot(freq[n//2:n], (np.hypot(spec2.real, spec2.imag)/n*2.0)[n//2:n], '-.', color='tab:orange')
ax.set_xlabel("$f$, отн. ед.", fontsize=10)
ax.set_ylabel("$U(f)$, В", fontsize=10)
ax.legend(loc='best')
if not full_scale:
    ax.set_xlim(0, 3.5*f)

fig, ax = plt.subplots()
if not use_only_fft:
    ax.plot(t, sig_dft.real, label='ОДПФ')
ax.plot(t, sig_fft.real, '--', label='ОБПФ')
ax.plot(t, u, '-.', label='Исх. сигнал')
ax.set_xlabel("$t$, отн. ед.", fontsize=10)
ax.set_ylabel("$U(t)$, В", fontsize=10)
ax.legend(loc='best')
if not full_scale:
    ax.set_xlim(0, 1/f)

plt.show()