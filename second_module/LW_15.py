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

nch = 3
npp = 50
fc = np.array([1.0, 2.0, 3.0]) #Частоты первичных гармонических сигналов в каналах 1, 2, 3 [кГц]

T = float(input('Временной интервал, мс '))
n = int(8*T)*npp*nch #int(input('Число временных отсчетов (должно быть не менее {0: .0f}) '.format(8*T*npp)))

#Массивы первичных сигналов, поступающих на вход каналов 1, 2, 3
sig1 = [0] * n
sig2 = [0] * n
sig3 = [0] * n

#ПППИ (periodic sequence of rectangular pulses)
pcrp = [0] * n

#Массивы модулированных первичных, канальных и группового сигналов в тракте передачи
mch1 = [0] * n
mch2 = [0] * n
mch3 = [0] * n

#Массив моментов времени для отсчетов сигналов
time = [0] * n

#Шаг дискретизации по времени
h = T / (n-1)

#Формирование первичных сигналов в каналах
for i in range(n):
    time[i] = i*h
    sig1[i] = 1.0*cos(f2w(fc[0]-0.2)*time[i])+1.5*cos(f2w(fc[0])*time[i])+2.0*cos(f2w(fc[0]+0.2)*time[i]) if not pulse else 20*wb_pulse(time[i], T, 0.3, 2.0)
    sig2[i] = 2.0*cos(f2w(fc[1]-0.2)*time[i])+2.5*cos(f2w(fc[1])*time[i])+3.0*cos(f2w(fc[1]+0.2)*time[i]) if not pulse else 20*wb_pulse(time[i], T, 1.3, 3.0)
    sig3[i] = 1.5*cos(f2w(fc[2]-0.2)*time[i])+2.0*cos(f2w(fc[2])*time[i])+2.5*cos(f2w(fc[2]+0.2)*time[i]) if not pulse else 20*wb_pulse(time[i], T, 2.3, 3.4)

#Формирование канальных сигналов
for i in range(n//nch//npp):
    for j in range(nch):
        for k in range(npp):
            indx = i*nch*npp+j*npp+k
            pcrp[indx] = 1.0 if k < 0.75*npp else 0 #Формирование ПППИ
            #Канальные электронные ключи
            if j==0:
                mch1[indx] = pcrp[indx]*sig1[indx]
            elif j==1:
                mch2[indx] = pcrp[indx]*sig2[indx]
            elif j==2:
                mch3[indx] = pcrp[indx]*sig3[indx]


#Формирование группового сигнала (объединение канальных сигналов)
tgrp = np.array(np.array(mch1) + np.array(mch2) + np.array(mch3))

#Расчет спектра сигалов
sp_freq = np.fft.fftfreq(n, T/n)

sp_sig1 = np.fft.fft(sig1)
sp_sig2 = np.fft.fft(sig2)
sp_sig3 = np.fft.fft(sig3)

sp_pcrp = np.fft.fft(pcrp)

sp_mch1 = np.fft.fft(mch1)
sp_mch2 = np.fft.fft(mch2)
sp_mch3 = np.fft.fft(mch3)

sp_tgrp = np.fft.fft(tgrp)

#Спектры исходных первичных сигналов на входе СП ЧРК
sp_sig1 = np.hypot(sp_sig1.real, sp_sig1.imag)/n*2
sp_sig2 = np.hypot(sp_sig2.real, sp_sig2.imag)/n*2
sp_sig3 = np.hypot(sp_sig3.real, sp_sig3.imag)/n*2

sp_pcrp = np.hypot(sp_pcrp.real, sp_pcrp.imag)/n*2

#Спектры модулированных сигналов в каналах 1, 2, 3
sp_mch1 = np.hypot(sp_mch1.real, sp_mch1.imag)/n*2
sp_mch2 = np.hypot(sp_mch2.real, sp_mch2.imag)/n*2
sp_mch3 = np.hypot(sp_mch3.real, sp_mch3.imag)/n*2

#Спектр группового сигнала на выъходе тракта передачи и входе в тракт приема
sp_tgrp = np.hypot(sp_tgrp.real, sp_tgrp.imag)/n*2

#Построение графиков сигналов и их спектров в передающем тракте
fig, (ax0, ax1) = plt.subplots(nrows=2)
ax0.set_title('Распределения первичных сигналов')
ax0.plot(time, sig1, 'tab:blue', lw =1, label='$Ch_1$')
ax0.plot(time, sig2, 'tab:orange', lw =1, label='$Ch_2$')
ax0.plot(time, sig3, 'tab:green', lw =1, label='$Ch_3$')
if auto_scale:
    ax0.set_xlim(T/2-1, T/2+1)
else:
    ax0.set_xlim(time[0], time[-1])
ax0.set_xlabel("$t$, мс", fontsize=10)
ax0.set_ylabel("$V_{1,2,3}$, В", fontsize=10)
ax0.legend(loc='best')
ax1.set_title('Спектры первичных сигналов')
ax1.plot(sp_freq[0:n//2], sp_sig1[0:n//2], 'tab:blue', lw =1, label='$Ch_1$')
ax1.plot(sp_freq[0:n//2], sp_sig2[0:n//2], 'tab:orange', lw =1, label='$Ch_2$')
ax1.plot(sp_freq[0:n//2], sp_sig3[0:n//2], 'tab:green', lw =1, label='$Ch_3$')
ax1.set_xlim(0, 5)
ax1.set_xlabel("$f$, кГц", fontsize=10)
ax1.set_ylabel("$V_{1,2,3}$, В", fontsize=10)
ax1.legend(loc='best')
fig.tight_layout()

fig, (ax0, ax1) = plt.subplots(nrows=2)
ax0.set_title('ПППИ')
ax0.plot(time, pcrp, 'tab:blue', lw =1)
if auto_scale:
    ax0.set_xlim(T/2-1, T/2+1)
else:
    ax0.set_xlim(time[0], time[-1])
ax0.set_xlabel("$t$, мс", fontsize=10)
ax0.set_ylabel("$V_{1,2,3}$, В", fontsize=10)
ax1.set_title('Спектр ПППИ')
ax1.plot(sp_freq[0:n//2], sp_pcrp[0:n//2], 'tab:blue', lw =1)
#ax1.set_xlim(0, 25)
ax1.set_xlabel("$f$, кГц", fontsize=10)
ax1.set_ylabel("$V_{1,2,3}$, В", fontsize=10)
fig.tight_layout()

fig, (ax0, ax1) = plt.subplots(nrows=2)
ax0.set_title('Распределения модулированных сигналов')
ax0.plot(time, mch1, 'tab:blue', lw =1, label='$Ch_1$')
ax0.plot(time, mch2, 'tab:orange', lw =1, label='$Ch_2$')
ax0.plot(time, mch3, 'tab:green', lw =1, label='$Ch_3$')
if auto_scale:
    ax0.set_xlim(T/2-1, T/2+1)
else:
    ax0.set_xlim(time[0], time[-1])
ax0.set_xlabel("$t$, мс", fontsize=10)
ax0.set_ylabel("$V_{1,2,3}$, В", fontsize=10)
ax0.legend(loc='best')
ax1.set_title('Спектры модулированных сигналов')
ax1.plot(sp_freq[0:n//2], sp_mch1[0:n//2], 'tab:blue', lw =1, label='$Ch_1$')
ax1.plot(sp_freq[0:n//2], sp_mch2[0:n//2], 'tab:orange', lw =1, label='$Ch_2$')
ax1.plot(sp_freq[0:n//2], sp_mch3[0:n//2], 'tab:green', lw =1, label='$Ch_3$')
ax1.set_xlim(0, 25)
ax1.set_xlabel("$f$, кГц", fontsize=10)
ax1.set_ylabel("$V_{1,2,3}$, В", fontsize=10)
ax1.legend(loc='best')
fig.tight_layout()

fig, (ax0, ax1) = plt.subplots(nrows=2)
ax0.set_title('Групповой сигнал')
ax0.plot(time, tgrp, 'tab:blue', lw =1)
if auto_scale:
    ax0.set_xlim(T/2-1, T/2+1)
else:
    ax0.set_xlim(time[0], time[-1])
ax0.set_xlabel("$t$, мс", fontsize=10)
ax0.set_ylabel("$V_{1,2,3}$, В", fontsize=10)
ax1.set_title('Спектр группового сигнала')
ax1.plot(sp_freq[0:n//2], sp_tgrp[0:n//2], 'tab:blue', lw =1)
ax1.set_xlim(0, 25)
ax1.set_xlabel("$f$, кГц", fontsize=10)
ax1.set_ylabel("$V_{1,2,3}$, В", fontsize=10)
fig.tight_layout()
#plt.show()

#Разделение группового сигнала на канальные сигналы
for i in range(n//nch//npp):
    for j in range(nch):
        for k in range(npp):
            indx = i*nch*npp+j*npp+k
            mch1[indx] = 0
            mch2[indx] = 0
            mch3[indx] = 0
            #Канальные электронные ключи
            if j==0:
                mch1[indx] = tgrp[indx]
            elif j==1:
                mch2[indx] = tgrp[indx]
            elif j==2:
                mch3[indx] = tgrp[indx]

#Канальные ФНЧ для выделения первичных сигналов
rsig1 = filter(time, mch1, 0.3, 3.4).real
rsig2 = filter(time, mch2, 0.3, 3.4).real
rsig3 = filter(time, mch3, 0.3, 3.4).real

#Спектры канальных сигналов после разделения
sp_mch1 = np.fft.fft(mch1)
sp_mch2 = np.fft.fft(mch2)
sp_mch3 = np.fft.fft(mch3)
sp_mch1 = np.hypot(sp_mch1.real, sp_mch1.imag)/n*2
sp_mch2 = np.hypot(sp_mch2.real, sp_mch2.imag)/n*2
sp_mch3 = np.hypot(sp_mch3.real, sp_mch3.imag)/n*2

#Спектры первичных сигналов, выделенных из канальных на выходе СП ЧРК
sp_rsig1 = np.fft.fft(rsig1)
sp_rsig2 = np.fft.fft(rsig2)
sp_rsig3 = np.fft.fft(rsig3)
sp_rsig1 = np.hypot(sp_rsig1.real, sp_rsig1.imag)/n*2
sp_rsig2 = np.hypot(sp_rsig2.real, sp_rsig2.imag)/n*2
sp_rsig3 = np.hypot(sp_rsig3.real, sp_rsig3.imag)/n*2

#Построение графиков сигналов и их спектров на приемном участке МСП ЧРК
fig, (ax0, ax1) = plt.subplots(nrows=2)
ax0.set_title('Распределения канальных сигналов в тракте приема')
ax0.plot(time, mch1, 'tab:blue', lw =1, label='$Ch_1$')
ax0.plot(time, mch2, 'tab:orange', lw =1, label='$Ch_2$')
ax0.plot(time, mch3, 'tab:green', lw =1, label='$Ch_3$')
if auto_scale:
    ax0.set_xlim(T/2-1, T/2+1)
else:
    ax0.set_xlim(time[0], time[-1])
ax0.set_xlabel("$t$, мс", fontsize=10)
ax0.set_ylabel("$V_{1,2,3}$, В", fontsize=10)
ax0.legend(loc='best')
ax1.set_title('Спектры канальных сигналов в тракте приема')
ax1.plot(sp_freq[0:n//2], sp_mch1[0:n//2], 'tab:blue', lw =1, label='$Ch_1$')
ax1.plot(sp_freq[0:n//2], sp_mch2[0:n//2], 'tab:orange', lw =1, label='$Ch_2$')
ax1.plot(sp_freq[0:n//2], sp_mch3[0:n//2], 'tab:green', lw =1, label='$Ch_3$')
ax1.set_xlim(0, 25)
ax1.set_xlabel("$f$, кГц", fontsize=10)
ax1.set_ylabel("$V_{1,2,3}$, В", fontsize=10)
ax1.legend(loc='best')
fig.tight_layout()

fig, (ax0, ax1) = plt.subplots(nrows=2)
ax0.set_title('Распределения первичных сигналов')
ax0.plot(time, rsig1, 'tab:blue', lw =1, label='$Ch_1$')
ax0.plot(time, rsig2, 'tab:orange', lw =1, label='$Ch_2$')
ax0.plot(time, rsig3, 'tab:green', lw =1, label='$Ch_3$')
if auto_scale:
    ax0.set_xlim(T/2-1, T/2+1)
else:
    ax0.set_xlim(time[0], time[-1])
ax0.set_xlabel("$t$, мс", fontsize=10)
ax0.set_ylabel("$V_{1,2,3}$, В", fontsize=10)
ax0.legend(loc='best')
ax1.set_title('Спектры первичных сигналов')
ax1.plot(sp_freq[0:n//2], sp_rsig1[0:n//2], 'tab:blue', lw =1, label='$Ch_1$')
ax1.plot(sp_freq[0:n//2], sp_rsig2[0:n//2], 'tab:orange', lw =1, label='$Ch_2$')
ax1.plot(sp_freq[0:n//2], sp_rsig3[0:n//2], 'tab:green', lw =1, label='$Ch_3$')
ax1.set_xlim(0, 5)
ax1.set_xlabel("$f$, кГц", fontsize=10)
ax1.set_ylabel("$V_{1,2,3}$, В", fontsize=10)
ax1.legend(loc='best')
fig.tight_layout()
plt.show()