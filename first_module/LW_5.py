import numpy as np
from math import *
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


# Исходный сигнал
def signal1(x):
    return np.sin(2.0 * pi * x) + 2.0 * np.cos(4.0 * pi * x) + 0.5 * np.cos(6.0 * pi * x)


# Сигнал на выходе фильтра
def signal2(x):
    return (np.sin(2.0 * pi * x) * np.exp(-Gam(f, L, C1, C2).real) +
            2.0 * np.cos(4.0 * pi * x) * np.exp(-Gam(2.0 * f, L, C1, C2).real) +
            0.5 * np.cos(6.0 * pi * x) * np.exp(-Gam(3.0 * f, L, C1, C2).real))


# f->w
def f2w(f):
    return 2.0 * pi * f


def Z(f, L, C2):
    return (1.0 - f2w(f) ** 2 * L * C2) / (f2w(f) * C2 * 1j)


def Y(f, C1):
    return 0.5 * f2w(f) * C1 * 1j


def Gam(f, L, C1, C2):
    ZY = Z(f, L, C2) * Y(f, C1)
    return 2.0 * np.arcsinh(np.sqrt(ZY))


def filter(time, signal, fl, fh):
    n = len(signal)
    freq = np.fft.fftfreq(n, time[1] - time[0])
    spectr = np.fft.fft(signal)
    for i in range(n):
        if not fl <= abs(freq[i]) <= fh:
            spectr[i] *= 0 + 0j
    return np.fft.ifft(spectr)


f = float(input('Опорная частота сигнала '))
T = float(input('Временной интервал '))
n = int(input('Число временных отсчетов '))

fl = float(input('Нижняя граничная частота фильтра (fl > 0) '))
fh = float(input('Верхняя граничная частота фильтра '))
L = float(input('Индуктивность фильтра '))

C1 = 2.0 / L / (f2w(fh) ** 2 - f2w(fl) ** 2)
C2 = 1.0 / (f2w(fl) ** 2 * L)
freq = np.linspace(0, fh * 1.2, n)

print('Параметры фильтра:')
print('C1 = ', C1)
print('C2 = ', C2)
print('L = ', L)

Gama = Gam(freq, L, C1, C2)

print('Затухание сигнала:')
print('A(', 1 * f, ') = ', Gam(f, L, C1, C2).real * 8.686, 'дБ')
print('A(', 2 * f, ') = ', Gam(2 * f, L, C1, C2).real * 8.686, 'дБ')
print('A(', 3 * f, ') = ', Gam(3 * f, L, C1, C2).real * 8.686, 'дБ')

print('Амплитуды гармонических составляющих после прохождения через фильтр:')
print('U(', 1 * f, ') = ', 1.0 * 10 ** (-0.05 * Gam(f, L, C1, C2).real * 8.686), 'В')
print('U(', 2 * f, ') = ', 2.0 * 10 ** (-0.05 * Gam(2 * f, L, C1, C2).real * 8.686), 'В')
print('U(', 3 * f, ') = ', 0.5 * 10 ** (-0.05 * Gam(3 * f, L, C1, C2).real * 8.686), 'В')

plt.plot(freq, Gama.real * 8.686, color='tab:blue', label=r'$\alpha(f), дБ$')
plt.tick_params(axis='y', color='tab:blue')
plt.legend(loc='lower right')
plt.twinx()
plt.plot(freq, Gama.imag, color='tab:red', label=r'$\varphi(f), рад.$')
plt.tick_params(axis='y', labelcolor='tab:red')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

t = np.linspace(0, T, n)
uinp0 = signal1(f * t)
uout1 = signal2(f * t)
uout2 = filter(t, uinp0, fl, fh).real

plt.plot(t, uinp0, label='$U_{вх}(t)$')
plt.plot(t, uout1, '--', label='$U_{вых}(t) - LC-фильтр$')
plt.plot(t, uout2, '-.', label='$U_{вых}(t) - БПФ-фильтр$')
plt.axis(xmin=0, xmax=2 / f)
plt.legend(loc='best')
plt.tight_layout()
plt.show()

sp_inp = np.abs(np.fft.fft(uinp0)) / n * 2.0
sp_inp[0] *= 0.5
sp_out = np.abs(np.fft.fft(uout1)) / n * 2.0
sp_out[0] *= 0.5
freq = np.fft.fftfreq(n, T / n)

plt.plot(freq, sp_inp, '-.', label='$U_{вх}(f)$')
plt.plot(freq, sp_out, '--', label='$U_{вых}(f)$')
plt.axis(xmin=0, xmax=4 * f)
plt.legend(loc='best')
plt.tight_layout()
plt.show()
