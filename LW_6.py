import numpy as np
from math import *
import matplotlib.pyplot as plt


# Гармонический сигнал на входе фильтра
def signal1(x):
    return 1.0 + np.sin(2.0 * pi * x) + 2.0 * np.cos(4.0 * pi * x) + 0.5 * np.cos(6.0 * pi * x)


# Гармонический сигнал на выходе фильтра
def signal2(x):
    return (1.0 * np.exp(-Gam(0.001, L, C1, C2)) + np.sin(2.0 * pi * x) * np.exp(-Gam(f, L, C1, C2)) +
            2.0 * np.cos(4.0 * pi * x) * np.exp(-Gam(2.0 * f, L, C1, C2)) +
            0.5 * np.cos(6.0 * pi * x) * np.exp(-Gam(3.0 * f, L, C1, C2)))


# Широкополосный импульс на входе фильтра
def ipulse(t, Tc, fn, fv):
    freq = (fv + fn) * 0.5
    dt = 1.0 / (fv - fn)
    return np.exp(-(0.5 * Tc - t) ** 2 / dt ** 2 * 0.5) * np.sin(2.0 * pi * freq * t)


# Широкополосный импульс на выходе фильтра
def opulse(t, Tc, fn, fv, L, C1, C2):
    signal = ipulse(t, Tc, fn, fv)
    n = len(signal)
    freq = np.fft.fftfreq(n, t[1] - t[0])
    spectr = np.fft.fft(signal)
    for i in range(n):
        if freq[i] != 0:
            spectr[i] *= np.exp(-Gam(abs(freq[i]), L, C1, C2))
    return np.fft.ifft(spectr)


def f2w(f):
    return 2.0 * pi * f


def Z1(f, C1):
    return 2.0 / (1j * f2w(f) * C1)


def Z2(f, C2):
    return 1.0 / (1j * f2w(f) * C2)


def Z3(f, L):
    return 1.0j * f2w(f) * L


def Gam(f, L, C1, C2):
    ZY = (Z2(f, C2) + Z3(f, L)) / Z1(f, C1)
    return 2.0 * np.arcsinh(np.sqrt(ZY))


# Характеристическое сопротивление фильтра
def Zw(f, L, C1, C2):
    return np.sqrt((Z1(f, C1) ** 2 * (Z2(f, C2) + Z3(f, L))) / (2 * Z1(f, C1) + Z2(f, C2) + Z3(f, L)))


pulse = False

f = float(input('Опорная частота сигнала '))
T = float(input('Временной интервал '))
n = int(input('Число временных отсчетов '))

fl = float(input('Нижняя граничная частота фильтра '))
fh = float(input('Верхняя граничная частота фильтра '))
f0 = (fl + fh) * 0.5
Z0 = float(input('Характеристическое сопротивление фильтра на частоте ' + str(f0) + ' '))

L = (sqrt(Z0 ** 2 * f2w(f0) ** 2 * (2 * f2w(fh) ** 2 - f2w(fl) ** 2 - f2w(f0) ** 2) /
          ((f2w(fh) ** 2 - f2w(fl) ** 2) ** 2 * (f2w(f0) ** 2 - f2w(fl) ** 2))))
C1 = 2.0 / L / (f2w(fh) ** 2 - f2w(fl) ** 2)
C2 = 1.0 / (f2w(fl) ** 2 * L)

print('Параметры фильтра:')
print('C1 = {0: f}\nC2 = {1: f}\nL = {2: f}'.format(C1, C2, L))

freq = np.linspace(0.8 * fl, fh * 1.2, n)

Gama = Gam(freq, L, C1, C2)
Zw = Zw(freq, L, C1, C2)
dF = (Gam(freq + 0.01, L, C1, C2).imag - Gam(freq - 0.01, L, C1, C2).imag) / 0.02

# Построение графиков
plt.plot(freq, Gama.real, color='tab:blue', label=r'$\alpha(f)$')
plt.tick_params(axis='y', labelcolor='tab:blue')
plt.legend(loc='lower right')
plt.twinx()
plt.plot(freq, Gama.imag, color='tab:orange', label=r'$\varphi(f)$')
plt.tick_params(axis='y', labelcolor='tab:orange')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

plt.plot(freq, abs(Zw), label='$|Z_0|(f)$')
plt.plot(freq, Zw.real, label='$Re(Z_0)(f)$')
plt.plot(freq, Zw.imag, label='$Im(Z_0)(f)$')
plt.vlines(f0, 0, Z0, color='tab:olive', linestyles='dashdot', lw=1)
plt.hlines(Z0, freq[0], f0, color='tab:olive', linestyles='dashdot', lw=1)
plt.legend(loc='best')
plt.tight_layout()
plt.show()

plt.plot(freq, dF, label='$b(f)$')
plt.legend(loc='lower right')
plt.twinx()
plt.plot(freq, np.exp(-Gama.real), color='tab:orange', label=r'$K(f)$')
plt.tick_params(axis='y', labelcolor='tab:orange')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

t = np.linspace(0, T, n)
uinp = ipulse(t, T, 0.5 * fl, 1.5 * fh) if pulse else signal1(f * t)
uout = opulse(t, T, 0.5 * fl, 1.5 * fh, L, C1, C2) if pulse else signal2(f * t)

plt.plot(t, uinp, label='$U_{вх}(t)$')
plt.plot(t, np.array(uout, dtype=complex).real, '--', label='$Re(U_{вых}(t))$')
plt.plot(t, np.array(uout, dtype=complex).imag, '-.', label='$Im(U_{вых}(t))$')
if pulse:
    plt.axis(xmin=T / 2 - 4 / (fh - fl), xmax=T / 2 + 4 / (fh - fl))
else:
    plt.axis(xmin=0, xmax=2 / f)
plt.legend(loc='best')
plt.tight_layout()
plt.show()

sp_inp = np.fft.fft(uinp)
sp_out = np.fft.fft(uout)
freq = np.fft.fftfreq(n, T / n)

plt.plot(freq[0:n // 2], (np.hypot(sp_inp.real, sp_inp.imag) / n * 2.0)[0:n // 2], '-', label='$U_{вх}(f)$')
plt.plot(freq[0:n // 2], (np.hypot(sp_out.real, sp_out.imag) / n * 2.0)[0:n // 2], '--', label='$U_{вых}(f)$')
plt.axis(xmin=0, xmax=4 * f)
plt.legend(loc='best')
plt.tight_layout()
plt.show()

if pulse:
    plt.plot(freq[0:n // 2], (np.hypot(sp_out.real, sp_out.imag) / np.hypot(sp_inp.real, sp_inp.imag))[0:n // 2], '--',
             label='$K(f)$')
    plt.axis(xmin=0, xmax=4 * f)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()