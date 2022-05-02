import numpy as np
from math import *
import matplotlib.pyplot as plt


# Гармонический сигнал
def harm_signal(t):
    return sin(2.0 * pi * fc * t)


# Производная гармонического сигнала
def d_harm_signal(t):
    return 2.0 * pi * fc * cos(2.0 * pi * fc * t)


# Короткий широкополосный импульс
def wb_pulse(t):
    freq = (1.5 * fh + 0.5 * fl) * 0.5
    dt = 1.0 / (1.5 * fh - 0.5 * fl)
    return exp(-(0.5 * Tc - t) ** 2 / dt ** 2 * 0.5) * sin(2.0 * pi * freq * t)


# Производная широкополосного импульса
def d_wb_pulse(t):
    freq = (1.5 * fh + 0.5 * fl) * 0.5
    dt = 1.0 / (1.5 * fh - 0.5 * fl)
    a1 = 2.0 * pi * freq * t * exp(-(0.5 * Tc - t) ** 2 / dt ** 2 * 0.5) * cos(2.0 * pi * freq * t)
    a2 = -2.0 * t / (2.0 * dt ** 2) * exp(-(0.5 * Tc - t) ** 2 / dt ** 2 * 0.5) * sin(2.0 * pi * freq * t)
    return a1 + a2


# Сигнал возбуждения ЛП
def signal(t):
    return wb_pulse(t) if pulse else harm_signal(t)


# Производная возбуждающего сигнала
def d_signal(t):
    return d_wb_pulse(t) if pulse else d_harm_signal(t)


# Перевод частоты в циклическую
def f2w(f):
    return 2.0 * pi * f


def Z1(f, C1):
    return 2.0 / (1j * f2w(f) * C1)


def Z2(f, C2):
    return 1.0 / (1j * f2w(f) * C2)


def Z3(f, L):
    return 1.0j * f2w(f) * L


# Постоянная распространения отдельной ячейки
def Gam(f, L, C1, C2):
    ZY = (Z2(f, C2) + Z3(f, L)) / Z1(f, C1)
    return 2.0 * np.arcsinh(np.sqrt(ZY))


# Характеристическое сопротивление отдельной ячейки
def Zw(f, L, C1, C2):
    return np.sqrt(((0.5 * Z1(f, C1)) ** 2 * (Z2(f, C2) + Z3(f, L))) / (2 * 0.5 * Z1(f, C1) + Z2(f, C2) + Z3(f, L)))


# Функции правых частей уравнений возбуждения ЛП
def d2V1():
    return (1.0 / (L * C1) * (aV[1] - aV[0] + aU[0]) + 1.0 / (Z0 * K0 * C1) * (A0 * d_signal(time[it]) - dV[0]))


def d2VN():
    return (1.0 / (L * C1) * (aV[Nc - 1] - aV[Nc] - aU[Nc - 1]) + 1.0 / (Z0 * KN * C1) * (
                AN * d_signal(time[it]) - dV[Nc]))


def d2Vs():
    return (0.5 / (L * C1) * (aV[ic - 1] - 2.0 * aV[ic] + aV[ic + 1] + aU[ic] - aU[ic - 1]))


def d2Us():
    return (1.0 / (L * C2) * (aV[ic] - aV[ic + 1] - aU[ic]) - G / C2 * dU[ic])


fc = float(input('Частота сигнала возбуждения ЛП '))
Tc = float(input('Временной интервал '))

fl = float(input('Нижняя граничная частота ЛП '))
fh = float(input('Верхняя граничная частота ЛП '))

Z0 = float(input('Характеристическое сопротивление одного звена ЛП на частоте сигнала ' + str(fc) + ' '))
Nc = int(input('Число ячеек в ЛП '))

pulse = False  # Включает импульсный сигнал, иначе гармонический

L = (sqrt(4 * Z0 ** 2 * f2w(fc) ** 2 * (f2w(fh) ** 2 - f2w(fc) ** 2) /
          ((f2w(fh) ** 2 - f2w(fl) ** 2) ** 2 * (f2w(fc) ** 2 - f2w(fl) ** 2))))
C1 = 2.0 / L / (f2w(fh) ** 2 - f2w(fl) ** 2)
C2 = 1.0 / (f2w(fl) ** 2 * L)
G = 0

npp = 50
dt = 1 / (fh * npp)
num = int(Tc / dt)  # Число временных отсчетов (шаг расчета уравнений возбуждения ЛП)

print('Параметры отдельной ячейки ЛП:')
print('C1 = {0: f}\nC2 = {1: f}\nL = {2: f}'.format(C1, C2, L))

freq = np.linspace(0.8 * fl, fh * 1.2, num)

Gama = Gam(freq, L, C1, C2)
Zw = Zw(freq, L, C1, C2)
dF = (Gam(freq + 0.005, L, C1, C2).imag - Gam(freq - 0.005, L, C1, C2).imag) / 0.01

# Решение уравнений возбуждения линии передачи

A0 = 1  # Амплитуда сигнала слева
AN = 0  # Амплитуда сигнала справа
K0 = KN = 1  # Коэффициенты при нагрузочных сопротивлениях

aU = [0] * Nc  # Массив напряжений на емкости C2
dU = [0] * Nc  # Массив производных напряжений на емкости C2
aV = [0] * (Nc + 1)  # Массив напряжений на емкости C1
dV = [0] * (Nc + 1)  # Массив производных напряжений на емкости C1

Vinp = [0] * num  # Массив входных напряжений
Vout = [0] * num  # Массив выходных напряжений
time = [0] * num  # Массив временных отсчетов

for it in range(num):
    time[it] = dt * it
    dV[0] += d2V1() * dt
    for ic in range(Nc):
        dU[ic] += d2Us() * dt
        if ic == 0:
            continue
        dV[ic] += d2Vs() * dt
    dV[Nc] += d2VN() * dt

    for ic in range(Nc):
        aV[ic] += dV[ic] * dt
        aU[ic] += dU[ic] * dt
    aV[Nc] += dV[Nc] * dt

    Vinp[it] = aV[0]
    Vout[it] = aV[Nc]
    if it % 100 == 0:
        print('{0: 7.3f} {1: 7.3f} {2: 7.3f} '.format(time[it], Vinp[it], Vout[it]))
spectr_inp = np.fft.fft(Vinp)
spectr_out = np.fft.fft(Vout)
fft_freq = np.fft.fftfreq(num, Tc / num)

# Графическая часть
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
plt.vlines(fc, 0, Z0, color='tab:olive', linestyles='dashdot', lw=1)
plt.hlines(Z0, freq[0], fc, color='tab:olive', linestyles='dashdot', lw=1)
plt.axis(ymin=-2 * Z0, ymax=2 * Z0)
plt.legend(loc='best')
plt.tight_layout()
plt.show()

plt.plot(freq, dF)
plt.tight_layout()
plt.show()

plt.plot(time, Vinp, label='$V_{inp}$')
plt.plot(time, Vout, label='$V_{out}$')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

vf_inp = np.abs(spectr_inp) / num * 2
vf_out = np.abs(spectr_out) / num * 2
plt.plot(fft_freq[0:num // 2], vf_inp[0:num // 2], label='$V_{inp}$')
plt.plot(fft_freq[0:num // 2], vf_out[0:num // 2], label='$V_{out}$')
plt.axis(xmin=-0.5, xmax=2 * fh)
plt.legend(loc='best')
plt.tight_layout()
plt.show()

if pulse:
    plt.plot(fft_freq[0:num // 2], vf_out[0:num // 2] / vf_inp[0:num // 2],
             label=r"$K(f)=\frac{U_{вых}(f)}{U_{вх}(f)}$")
    plt.axis(xmin=0, xmax=1.5 * fh)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()