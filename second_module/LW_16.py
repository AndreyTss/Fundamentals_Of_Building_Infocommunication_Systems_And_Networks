import numpy as np
from math import *
import matplotlib.pyplot as plt

def f2w(f):
    return 2.0*pi*f

#Короткий широкополосный импульс
def wb_pulse(t, Tc, fn, fv):
    freq = (fv + fn) * 0.5
    dt = 1.0 / (fv-fn)
    return exp(-(0.5*Tc-t)**2/dt**2*0.5)*sin(2.0*pi*freq*t)

def filter(time, signal, fl, fh):
    n = len(signal)
    freq = np.fft.fftfreq(n, time[1]-time[0])
    spectr = np.fft.fft(signal)
    for i in range(n):
        if not fl <= abs(freq[i]) <= fh:
            spectr[i] *= 0+0j
    return np.fft.ifft(spectr)

pulse = True
auto_scale = True #Автомасштабирование графиков сигналов по времени

nbt = 8  #8 бит
npp = 50
fc = 2.0 #Частота гармонического аналогового сигнала [кГц]

T = float(input('Временной интервал, мс '))
n = int(8*T)*npp*nbt #int(input('Число временных отсчетов (должно быть не менее {0: .0f}) '.format(8*T*npp)))

#АЦП
sig1 = [0] * n #Исходный аналоговый сигнал, поступающий на вход АЦП
sig2 = [0] * n #Дискретизированный сигнал
sig3 = [0] * n #Квантованный сигнал
sig4 = [0] * n #Кодированный (цифровой) сигнал

#ЦАП
sig5 = [0] * n #Квантованный сигнал
sig6 = [0] * n #Дискретизированый сигнал
sig7 = [0] * n #Импульсный дискретизированый сигнал

#Массив моментов времени для отсчетов сигналов
time = [0] * n

#Шаг дискретизации по времени
h = T / (n-1)
smax = 0
smin = 0
#Формирование аналогового сигнала, его дискретизация и квантование по уровню
for i in range(n):
    time[i] = i*h
    sig1[i] = 1.0*cos(f2w(fc-0.2)*time[i])+1.5*cos(f2w(fc)*time[i])+2.0*cos(f2w(fc+0.2)*time[i]) if not pulse else wb_pulse(time[i], T, 0.3, 3.4)
    if smax < sig1[i]: smax = sig1[i]
    if smin > sig1[i]: smin = sig1[i]

#smin = -2
#smax = 2

#Дискретизация и квантование по уровню аналогового сигнала
for i in range(n//nbt//npp):
    for j in range(nbt*npp):
        k = i*nbt*npp+j
        if j == 0:
            sig2[k] = sig1[k]
            sig3[k] = int((sig2[k] - smin) / (smax - smin) * (2**nbt-1))
        if sig3[k] > 2**nbt: sig3[k] = 2**nbt-1
        if sig3[k] < 0: sig3[k] = 0

#Кодирование (формирование цифрового сигнала)
for i in range(n//nbt//npp):
    for j in range(nbt):
        bit = (int(sig3[i*nbt*npp])>>j) & 1
        for k in range(npp):
            sig4[i*nbt*npp+j*npp+k] = bit

#Преобразование цифрового сигнала в аналоговый
for i in range(n//nbt//npp):
    bit = 0
    for j in range(nbt):
        bit = int(sig4[i*nbt*npp+j*npp])
        sig5[i*nbt*npp] += bit << j
    sig6[i*nbt*npp] = smin + sig5[i*nbt*npp] / (2**nbt-1) * (smax-smin)
    for j in range(npp*nbt):
        indx = i*nbt*npp+j-nbt*npp//2
        if indx < 0: continue
        sig7[indx] = sig6[i*nbt*npp]

#Фильтрация дискретизированного импульсного сигнала (АИМ)
sig8 = filter(time, sig6, 0.3, 3.4)

#Построение графиков сигналов на этапе АЦП
fig, ax = plt.subplots()
ax.set_title('Исходный аналоговый сигнал')
ax.plot(time, sig1, 'tab:orange', lw =1)
if auto_scale:
    ax.set_xlim(T/2-1, T/2+1)
else:
    ax.set_xlim(time[0], time[-1])
ax.set_xlabel("$t$, мс", fontsize=10)
ax.set_ylabel("$V_{0}$, В", fontsize=10)
fig.tight_layout()

fig, ax = plt.subplots()
ax.set_title('Дискретизированный сигнал')
ax.plot(time, sig2, 'tab:green', lw =1)
if auto_scale:
    ax.set_xlim(T/2-1, T/2+1)
else:
    ax.set_xlim(time[0], time[-1])
ax.set_xlabel("$t$, мс", fontsize=10)
ax.set_ylabel("$V_{0}$, В", fontsize=10)
fig.tight_layout()

fig, ax = plt.subplots()
ax.set_title('Квантованный сигнал')
ax.plot(time, sig3, 'tab:olive', lw =1)
if auto_scale:
    ax.set_xlim(T/2-1, T/2+1)
else:
    ax.set_xlim(time[0], time[-1])
ax.set_xlabel("$t$, мс", fontsize=10)
ax.set_ylabel("$V_{0}$, В", fontsize=10)
fig.tight_layout()

fig, ax = plt.subplots()
ax.set_title('Кодированный (цифровой) сигнал')
ax.plot(time, sig4, 'tab:blue', lw =1)
if auto_scale:
    ax.set_xlim(T/2-1, T/2+1)
else:
    ax.set_xlim(time[0], time[-1])
ax.set_xlabel("$t$, мс", fontsize=10)
ax.set_ylabel("$V_{0}$, В", fontsize=10)
fig.tight_layout()
#plt.show()

#Построение графиков сигналов на этапе ЦАП
fig, ax = plt.subplots()
ax.set_title('Квантованный сигнал на этапе ЦАП')
ax.plot(time, sig5, 'tab:green', lw =1)
if auto_scale:
    ax.set_xlim(T/2-1, T/2+1)
else:
    ax.set_xlim(time[0], time[-1])
ax.set_xlabel("$t$, мс", fontsize=10)
ax.set_ylabel("$V_{0}$, В", fontsize=10)
fig.tight_layout()

fig, ax = plt.subplots()
ax.set_title('Дискретизированный сигнал')
ax.plot(time, sig6, 'tab:blue', lw =1)
if auto_scale:
    ax.set_xlim(T/2-1, T/2+1)
else:
    ax.set_xlim(time[0], time[-1])
ax.set_xlabel("$t$, мс", fontsize=10)
ax.set_ylabel("$V_{0}$, В", fontsize=10)
fig.tight_layout()

fig, ax = plt.subplots()
ax.set_title('Дискретизированный импульсный сигнал')
ax.plot(time, sig7, 'tab:red', lw =1)
ax.plot(time, sig6, 'tab:blue', lw =1, ls = '--', )
if auto_scale:
    ax.set_xlim(T/2-1, T/2+1)
else:
    ax.set_xlim(time[0], time[-1])
ax.set_xlabel("$t$, мс", fontsize=10)
ax.set_ylabel("$V_{0}$, В", fontsize=10)
fig.tight_layout()

fig, ax = plt.subplots()
ax.set_title('Аналоговый сигнал')
ax.plot(time, sig8.real, 'tab:blue', lw =1, label='АЦП/ЦАП')
ax.plot(time, sig1, 'tab:orange', lw =1, ls = '--', label='Исходный')
if auto_scale:
    ax.set_xlim(T/2-1, T/2+1)
else:
    ax.set_xlim(time[0], time[-1])
ax.set_xlabel("$t$, мс", fontsize=10)
ax.set_ylabel("$V_{0}$, В", fontsize=10)
ax.legend(loc='best')
fig.tight_layout()
plt.show()