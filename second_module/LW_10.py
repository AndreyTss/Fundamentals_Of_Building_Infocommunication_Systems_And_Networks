import numpy as np
from math import *
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


# Гармонический сигнал
def harm_signal(t, fc):
    coef = 1 if t > 1 else t  # 0.5*t**2
    return -coef * cos(2.0 * pi * fc * t)


# Производная гармонического сигнала
def d_harm_signal(t, fc):
    coef = 1 if t > 1 else t
    return 2.0 * pi * fc1 * coef * sin(2.0 * pi * fc * t)


# Широкополосный импульс
def wb_pulse(t, fn, fv):
    freq = (fv + fn) * 0.5
    dt = 1.0 / (fv - fn)
    return exp(-(0.5 * Tc - t) ** 2 / dt ** 2 * 0.5) * sin(2.0 * pi * freq * t)


# Производная широкополосного импульса
def d_wb_pulse(t, fn, fv):
    freq = (fv + fn) * 0.5
    dt = 1.0 / (fv - fn)
    a1 = 2.0 * pi * freq * t * exp(-(0.5 * Tc - t) ** 2 / dt ** 2 * 0.5) * cos(2.0 * pi * freq * t)
    a2 = -2.0 * t / (2.0 * dt ** 2) * exp(-(0.5 * Tc - t) ** 2 / dt ** 2 * 0.5) * sin(2.0 * pi * freq * t)
    return a1 + a2


# Перевод частоты в циклическую
def f2w(f):
    return 2.0 * pi * f


def fft_filter(time, signal, fl, fh):
    n = len(signal)
    freq = np.fft.fftfreq(n, time[1] - time[0])
    spectr = np.fft.fft(signal)
    for i in range(n):
        if not fl <= abs(freq[i]) <= fh:
            spectr[i] *= 0 + 0j
    return np.fft.ifft(spectr).real


def Z(f, L, C2):
    return (1.0 - f2w(f) ** 2 * L * C2) / (f2w(f) * C2 * 1j)


def Y(f, C1):
    return 0.5 * f2w(f) * C1 * 1j


def Gam(f, L, C1, C2):
    ZY = Z(f, L, C2) * Y(f, C1)
    return 2.0 * np.arcsinh(np.sqrt(ZY))


def lc_filter(time, signal, fn, fv, z0):
    # Расчет параметров фильтра
    f0 = 0.5 * (fn + fv)
    L = (sqrt(4.0 * z0 ** 2 * f2w(f0) ** 2 * (f2w(fv) ** 2 - f2w(f0) ** 2) /
              ((f2w(fv) ** 2 - f2w(fn) ** 2) ** 2 * (f2w(f0) ** 2 - f2w(fn) ** 2))))
    C1 = 2.0 / L / (f2w(fv) ** 2 - f2w(fn) ** 2)
    C2 = 1.0 / (f2w(fn) ** 2 * L)

    n = len(signal)
    freq = np.fft.fftfreq(n, time[1] - time[0])
    spectr = np.fft.fft(signal)
    for i in range(n):
        spectr[i] *= 0 if freq[i] == 0 else exp(-Gam(abs(freq[i]), L, C1, C2).real)
    return np.fft.ifft(spectr).real


ideal_filter = True
pulse1 = False
pulse2 = False

fc1 = 6  # Вещание в полосе частот [4..8]
fc2 = 14  # Вещание в полосе частот [12..16]
Tc = float(input('Временной интервал '))

fl = float(input('Нижняя граничная частота ЛП '))
fh = float(input('Верхняя граничная частота ЛП '))
f0 = (fc1 + fc2) * 0.5  # (fl + fh) * 0.5
Z0 = float(input('Характеристическое сопротивление одного звена ЛП на частоте ' + str(f0) + ' '))
Nc = int(input('Число ячеек в ЛП '))

L = (sqrt(4.0 * Z0 ** 2 * f2w(f0) ** 2 * (f2w(fh) ** 2 - f2w(f0) ** 2) /
          ((f2w(fh) ** 2 - f2w(fl) ** 2) ** 2 * (f2w(f0) ** 2 - f2w(fl) ** 2))))
C1 = 2.0 / L / (f2w(fh) ** 2 - f2w(fl) ** 2)
C2 = 1.0 / (f2w(fl) ** 2 * L)
G = 0

print('Параметры отдельной ячейки ЛП:')
print('C1 = {0: f}\nC2 = {1: f}\nL = {2: f}'.format(C1, C2, L))

npp = 10  # Количество точек на период гармонического сигнала
dt = 1 / (max(fc1, fc2) * npp)  # Шаг по времени
num = int(Tc / dt)  # Количество временных отсчетов

freq = np.linspace(0.8 * fl, fh * 1.2, num)

A0 = 1  # Амплитуда сигнала в направлении слева направо
AN = 1  # Амплитуда сигнала в направлении справа налево
K0 = KN = 1  # Коэффициенты при нагрузочных сопротивлениях

# Количество итераций для решения уравнений возбуждения
dpp = 50
print('dpp = {0: d}'.format(dpp))

aU = [0] * Nc  # Массив напряжений на емкости C2
dU = [0] * Nc  # Массив производных напряжений на емкости C2
aV = [0] * (Nc + 1)  # Массив напряжений на емкости C1
dV = [0] * (Nc + 1)  # Массив производных напряжений на емкости C1

Vinp = [0] * num  # Массив входных напряжений
Vout = [0] * num  # Массив выходных напряжений
time = [0] * num  # Массив временных отсчетов

Vs = [0] * npp  # Массив напряжений на C1 вдоль ЛП на одном периоде сигнала
for i in range(npp): Vs[i] = [0] * (Nc + 1)

# Решение уравнений возбуждения ЛП
for it in range(num):
    time[it] = dt * it
    for i in range(dpp):
        d_signal1 = d_wb_pulse(time[it], 4, 8) if pulse1 else d_harm_signal(time[it], fc1)
        dV[0] += (1.0 / (L * C1) * (aV[1] - aV[0] + aU[0]) + 1.0 / (Z0 * K0 * C1) * (A0 * d_signal1 - dV[0])) * dt / dpp
        for ic in range(Nc):
            dU[ic] += (1.0 / (L * C2) * (aV[ic] - aV[ic + 1] - aU[ic]) - G / C2 * dU[ic]) * dt / dpp
            if ic == 0: continue
            dV[ic] += (0.5 / (L * C1) * (aV[ic - 1] - 2.0 * aV[ic] + aV[ic + 1] + aU[ic] - aU[ic - 1])) * dt / dpp
        d_signal2 = d_wb_pulse(time[it], 12, 16) if pulse2 else d_harm_signal(time[it], fc2)
        dV[Nc] += (1.0 / (L * C1) * (aV[Nc - 1] - aV[Nc] - aU[Nc - 1]) + 1.0 / (Z0 * KN * C1) * (
                    AN * d_signal2 - dV[Nc])) * dt / dpp

        for ic in range(Nc):
            aV[ic] += dV[ic] * dt / dpp
            aU[ic] += dU[ic] * dt / dpp
        aV[Nc] += dV[Nc] * dt / dpp

    if num - it <= npp:
        for ic in range(Nc + 1):
            Vs[it - (num - npp)][ic] = aV[ic]

    Vinp[it] = aV[0]
    Vout[it] = aV[Nc]
    if it % 100 == 0:
        print('{0: 7.3f} {1: 7.3f} {2: 7.3f} '.format(time[it], Vinp[it], Vout[it]))

if ideal_filter:
    v_left = fft_filter(time, Vinp, 12, 16)
    v_right = fft_filter(time, Vout, 4, 8)
else:
    v_left = lc_filter(time, Vinp, 12, 16, Z0)
    v_right = lc_filter(time, Vout, 4, 8, Z0)

# Расчет спектра сигалов слева и справа
spectr_inp = np.abs(np.fft.fft(Vinp)) / num * 2
spectr_out = np.abs(np.fft.fft(Vout)) / num * 2
fft_freq = np.fft.fftfreq(num, Tc / num)

plt.plot(time, Vinp, time, Vout)
plt.show()

plt.plot(fft_freq[0:num // 2], spectr_inp[0:num // 2], label='$V_{inp}(f)$')
plt.plot(fft_freq[0:num // 2], spectr_out[0:num // 2], label='$V_{out}(f)$')
plt.axis(xmin=0, xmax=20)
plt.legend(loc='best')
plt.show()

cells = np.linspace(0, Nc, Nc + 1)
z_spl = np.linspace(0, Nc, (Nc + 1) * 10)
for i in range(npp):
    spl = make_interp_spline(cells, Vs[i], k=3)
    plt.plot(z_spl, spl(z_spl), label="t = {0: .3f} с".format(time[num - npp + i]), lw=1)
plt.legend(loc='best')
plt.show()

sig_l = [0] * num
sig_r = [0] * num
for i in range(num):
    sig_l[i] = 2.0 * wb_pulse(time[i], 4, 8) if pulse1 else 0.5 * harm_signal(time[i], fc1)
    sig_r[i] = 2.0 * wb_pulse(time[i], 12, 16) if pulse2 else 0.5 * harm_signal(time[i], fc2)

# plt.plot(time, sig_l, lw=1, label='$V_{исх}(t)$')
plt.plot(time, v_right, lw=1, label='$V_{справа}(t)$')
plt.show()

# plt.plot(time, sig_r, lw=1, label='$V_{исх}$')
plt.plot(time, v_left, lw=1, label='$V_{слева}$')
plt.show()

spectr_left = np.abs(np.fft.fft(v_left)) / num * 2
spectr_right = np.abs(np.fft.fft(v_right)) / num * 2
fft_freq = np.fft.fftfreq(num, Tc / num)
plt.plot(fft_freq[0:num // 2], spectr_left[0:num // 2], label='$V_{слева}(f)$')
plt.plot(fft_freq[0:num // 2], spectr_right[0:num // 2], '--', label='$V_{справа}(f)$')
plt.axis(xmin=0, xmax=20)
plt.legend(loc='best')
plt.show()