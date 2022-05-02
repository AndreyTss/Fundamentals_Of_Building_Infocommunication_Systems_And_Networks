import numpy as np
from math import *
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


# Производная гармонического сигнала
def d_harm_signal1(t):
    return 2.0 * pi * fc * sin(2.0 * pi * fc * t)


# Производная гармонического сигнала с плавным нарастанием
def d_harm_signal2(t):
    coef = 1 if t > 1 else t
    return 2.0 * pi * fc * coef * sin(2.0 * pi * fc * t)


# Перевод частоты в циклическую
def f2w(f):
    return 2.0 * pi * f


fc = float(input('Частота сигнала возбуждения ЛП '))
Tc = float(input('Временной интервал '))

fl = float(input('Нижняя граничная частота ЛП '))
fh = float(input('Верхняя граничная частота ЛП '))
f0 = fc if fl < fc < fh else (fl + fh) * 0.5
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
dt = 1 / (fc * npp)  # Шаг по времени
num = int(Tc / dt)  # Количество временных отсчетов

freq = np.linspace(0.8 * fl, fh * 1.2, num)

# Задание производной сигнала возбуждения ЛП
d_signal = d_harm_signal1

A0 = 1  # Амплитуда сигнала слева
AN = 0  # Амплитуда сигнала справа
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
        dV[0] += (1.0 / (L * C1) * (aV[1] - aV[0] + aU[0]) + 1.0 / (Z0 * K0 * C1) * (
                    A0 * d_signal(time[it]) - dV[0])) * dt / dpp
        for ic in range(Nc):
            dU[ic] += (1.0 / (L * C2) * (aV[ic] - aV[ic + 1] - aU[ic]) - G / C2 * dU[ic]) * dt / dpp
            if ic == 0: continue
            dV[ic] += (0.5 / (L * C1) * (aV[ic - 1] - 2.0 * aV[ic] + aV[ic + 1] + aU[ic] - aU[ic - 1])) * dt / dpp
        dV[Nc] += (1.0 / (L * C1) * (aV[Nc - 1] - aV[Nc] - aU[Nc - 1]) + 1.0 / (Z0 * KN * C1) * (
                    AN * d_signal(time[it]) - dV[Nc])) * dt / dpp

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

# Расчет спектра входного и выходного сигалов
spectr_inp = np.fft.fft(Vinp)
spectr_out = np.fft.fft(Vout)
fft_freq = np.fft.fftfreq(num, Tc / num)

plt.plot(time, Vinp, time, Vout)
plt.show()

sp_inp = np.abs(spectr_inp) / num * 2
sp_inp[0] *= 0.5
sp_out = np.abs(spectr_out) / num * 2
sp_out[0] *= 0.5
plt.plot(fft_freq[0:num // 2], sp_inp[0:num // 2], label='$V_{inp}$')
plt.plot(fft_freq[0:num // 2], sp_out[0:num // 2], label='$V_{out}$')
plt.axis(xmin=-1, xmax=1.1 * fh)
plt.legend(loc='best')
plt.show()

cells = np.linspace(0, Nc, Nc + 1)
z_spl = np.linspace(0, Nc, (Nc + 1) * 10)
for i in range(npp):
    spl = make_interp_spline(cells, Vs[i], k=3)
    plt.plot(z_spl, spl(z_spl), label="t = {0: .3f} с".format(time[num - npp + i]), lw=1)
plt.legend(loc='best')
plt.show()