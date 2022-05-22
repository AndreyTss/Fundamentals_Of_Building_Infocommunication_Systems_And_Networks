import numpy as np
from math import *
import matplotlib.pyplot as plt

def signal(x):
    return 1 + np.sin(2.0*pi*x) + 2.0 * np.cos(4.0*pi*x) + 0.5 * np.cos(6.0*pi*x)

def filter(time, signal, fl, fh):
    n = len(signal)
    freq = np.fft.fftfreq(n, time[1]-time[0])
    spectr = np.fft.fft(signal)
    for i in range(n):
        if not fl <= abs(freq[i]) <= fh:
            spectr[i] *= 0+0j
    return np.fft.ifft(spectr)

f = float(input('Частота 1-ой гармоники сигнала '))
T = float(input('Временной интервал '))
n = int(input('Число временных отсчетов '))
fl = float(input('Нижняя частота полосы пропускания фильтра '))
fh = float(input('Верхняя частота полосы пропускания фильтра '))

t = np.linspace(0, T, n)
u = signal(f*t)

print('Расчет БПФ...')
spec = np.fft.fft(u)
freq = np.fft.fftfreq(n, T/n)

print('Расчет ФНЧ...')
sig_lf = filter(t, u, 0, f)
print('Расчет ПФ...')
sig_pb = filter(t, u, f, 2*f)
print('Расчет ФВЧ...')
sig_hf = filter(t, u, 2*f, 3*f)
print('Расчет заданного фильтра...')
sig_uf = filter(t, u, fl, fh)

print('Расчет БПФ...')
spec_lf = np.fft.fft(sig_lf)
spec_pb = np.fft.fft(sig_pb)
spec_hf = np.fft.fft(sig_hf)
spec_uf = np.fft.fft(sig_uf)

fig, ax = plt.subplots()
ax.plot(t, u, color='tab:green')
ax.set_xlabel("$t$, отн. ед.", fontsize=10)
ax.set_ylabel("$U(t)$, В", fontsize=10)
ax.set_xlim(0, 1/f)

fig, ax = plt.subplots()
ax.plot(freq[0:n//2], (np.hypot(spec.real, spec.imag)/n*2.0)[0:n//2])
ax.set_xlabel("$f$, отн. ед.", fontsize=10)
ax.set_ylabel("$U(f)$, В", fontsize=10)
ax.set_xlim(-0.1*f, 4*f)

fig, ax = plt.subplots()
ax.plot(t, sig_lf.real, label='ФНЧ')
ax.plot(t, sig_pb.real, '--', label='ПФ')
ax.plot(t, sig_hf.real, '-.', label='ФВЧ')
ax.set_xlabel("$t$, отн. ед.", fontsize=10)
ax.set_ylabel("$U(t)$, В", fontsize=10)
ax.legend(loc='best')
ax.set_xlim(0, 1/f)

fig, ax = plt.subplots()
ax.plot(freq[0:n//2], (np.hypot(spec_lf.real, spec_lf.imag)/n*2.0)[0:n//2], '-', label='ФНЧ')
ax.plot(freq[0:n//2], (np.hypot(spec_pb.real, spec_pb.imag)/n*2.0)[0:n//2], '--', label='ПФ')
ax.plot(freq[0:n//2], (np.hypot(spec_hf.real, spec_hf.imag)/n*2.0)[0:n//2], '-.', label='ФВЧ')
ax.set_xlabel("$f$, отн. ед.", fontsize=10)
ax.set_ylabel("$U(f)$, В", fontsize=10)
ax.set_xlim(-0.1*f, 4*f)
ax.legend(loc='best')

fig, ax = plt.subplots()
ax.plot(t, sig_uf.real, label=r'${0:g} \leq f \leq {1:g}$'.format(fl, fh))
ax.set_xlabel("$t$, отн. ед.", fontsize=10)
ax.set_ylabel("$U(t)$, В", fontsize=10)
ax.legend(loc='best')
ax.set_xlim(0, 1/f)

fig, ax = plt.subplots()
ax.plot(freq[0:n//2], (np.hypot(spec_uf.real, spec_uf.imag)/n*2.0)[0:n//2], '-.', label=r'${0:g} \leq f \leq {1:g}$'.format(fl, fh))
ax.set_xlabel("$f$, отн. ед.", fontsize=10)
ax.set_ylabel("$U(f)$, В", fontsize=10)
ax.set_xlim(-0.1*f, 4*f)
ax.legend(loc='best')

plt.tight_layout()
plt.show()