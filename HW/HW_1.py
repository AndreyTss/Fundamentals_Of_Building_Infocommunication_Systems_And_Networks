from math import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import warnings

warnings.filterwarnings("ignore")

def lineplot(x_data, y_data, clr='#539caf', x_label="", y_label="", title=""):
    _, ax = plt.subplots()
    ax.plot(x_data, y_data, lw=2, color=clr, alpha=1)

    # Названия осей и заголовок графика
    ax.set_xlim(x_data[0], x_data[-1])
    # ax.set_ylim(min(y_data)*1.05, max(y_data)*1.05)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.show()


# Функция перевода частоты в циклическую
def f2w(f):
    return 2.0 * pi * f


# ВЧ импульс
def hf_pulse(t):
    ret = 0.0
    if (t >= Start and t <= Stop):
        ret = sin(2.0 * pi * fc * t)
    return ret


# Производная ВЧ импульса
def d_hf_pulse(t):
    ret = 0.0
    if (t >= Start and t <= Stop):
        ret = 2.0 * pi * fc * cos(2.0 * pi * fc * t)
    return ret


# НЧ импульс
def lf_pulse(t):
    ret = 1.0 if (t >= Start and t <= Stop) else 0.0
    if (t >= Start and t <= (Start + fwFront)):
        if (fwFront > 0.0):
            ret *= 0.5 * (1.0 - cos(pi * (t - Start) / fwFront))
    if (t >= (Stop - bwFront) and t <= Stop):
        if (bwFront > 0.0):
            ret *= 0.5 * (1.0 - cos(pi * (Stop - t) / bwFront))
    return ret


# Производная НЧ импульса
def d_lf_pulse(t):
    ret = 0.0
    if (t >= Start and t <= (Start + fwFront)):
        if (fwFront > 0.0):
            ret = 0.5 * sin(pi * (t - Start) / fwFront) * pi / fwFront
    if (t >= (Stop - bwFront) and t <= Stop):
        if (bwFront > 0.0):
            ret = -0.5 * sin(pi * (Stop - t) / bwFront) * pi / bwFront
    return ret


# Широкополосный импульс
def wb_pulse(t):
    freq = (fh + fl) * 0.5
    dt = 1.0 / (fh - fl)
    return exp(-(0.5 * Tc - t) ** 2 / dt ** 2 * 0.5) * sin(2.0 * pi * freq * t)


# Производная широкополосного импульса
def d_wb_pulse(t):
    freq = (1.5 * fh + 0.5 * fl) * 0.5
    dt = 1.0 / (1.5 * fh - 0.5 * fl)
    a1 = 2.0 * pi * freq * t * exp(-(0.5 * Time - t) ** 2 / dt ** 2 * 0.5) * cos(2.0 * pi * freq * t)
    a2 = -2.0 * t / (2.0 * dt ** 2) * exp(-(0.5 * Time - t) ** 2 / dt ** 2 * 0.5) * sin(2.0 * pi * freq * t)
    return a1 + a2


# Сигнал возбуждения линии передачи
def signal(t):
    if nsig == 0:
        return hf_pulse(t)
    elif nsig == 1:
        return hf_pulse(t) * lf_pulse(t)
    elif nsig == 2:
        return lf_pulse(t)
    elif nsig == 3:
        return 0.2 * (hf_pulse(t) * lf_pulse(t)) + lf_pulse(t)
    elif nsig == 4:
        return wb_pulse(t)
    else:
        return 0


# Производная сигнала возбуждения линии передачи
def d_signal(t):
    if nsig == 0:
        return d_hf_pulse(t)
    elif nsig == 1:
        return d_hf_pulse(t) * lf_pulse(t) + d_lf_pulse(t) * hf_pulse(t)
    elif nsig == 2:
        return d_lf_pulse(t)
    elif nsig == 3:
        return 0.2 * (d_hf_pulse(t) * lf_pulse(t) + d_lf_pulse(t) * hf_pulse(t)) + d_lf_pulse(t)
    elif nsig == 4:
        return d_wb_pulse(t)
    else:
        return 0


def Z1(f, C1):
    return 1.0 / (1.0j * f2w(f) * C1)


def Z2(f, C2, G):
    return 1.0 / ((1j * f2w(f) * C2) + G) if not lffilter else 0


def Z3(f, L):
    return 1.0j * f2w(f) * L


def Z(f, L, C2, G):
    return Z2(f, C2, G) + Z3(f, L)  # (1.0 - f2w(f)**2*L*C2)/(f2w(f)*C2*1j)


def Y(f, C1):
    return 1 / Z1(f, C1)  # f2w(f)*C1*1j


def Gam(f, L, C1, C2, G):
    ZY = Z(f, L, C2, G) * Y(f, C1)
    return 2.0 * np.arcsinh(np.sqrt(ZY))


# Характеристическое сопротивление отдельной ячейки
def Zw(f, L, C1, C2, G):
    return np.sqrt((Z1(f, C1) ** 2 * (Z2(f, C2, G) + Z3(f, L))) / (2 * Z1(f, C1) + Z2(f, C2, G) + Z3(f, L)))


# Начало программы
global Time, fwFront, bwFront, Start, Stop, fc, fl, fh, nsig, lffilter

# Данные для расчета
# lffilter - (True - ячека ЛП представляется звеном ФНЧ, False - ПФ)
# nvar     - Номер варианта
# nsig     - Сигнал возбуждения ЛП (0 - ВЧ сигнал, 1 - ВЧ импульс, 2 - НЧ импульс, 3 - НЧ+ВЧ импульс, 4 - ШП импульс)
# fl       - Нижняя граничная частота полосы пропускания ЛП
# fh       - Верхняя граничная частота полосы пропускания ЛП
# f0       - Частота для задания волнового сопротивления ЛП
# Z0       - Волновое сопротивление ЛП на частоте f0
# fc       - Опорная частота сигнала
# G        - Проводимость G
# K0       - Согласование на входе ЛП,  R0=K0*Z0 (K0=1-> R0=Z0)
# KN       - Согласование на выходе ЛП, RN=KN*Z0 (KN=1-> RN=Z0)
# A0       - Амплитуда возбуждающего сигнала на входе ЛП
# AN       - Амплитуда возбуждающего сигнала на выходе ЛП
# Time     - Временной интервал
# Nc       - Количество ячеек на длине ЛП
# fwFront  - Длительность переднего фронта импульсного сигнала
# bwFront  - Длительность заднего фронта импульсного сигнала
# Start    - Момент времени начала импульса
# Stop     - Время окончания импульса

show_tmp_graph = False
save_csv_files = False

ans = input("Использовать параметры по умолчанию? [y/n] ")
if not (ans == 'y' or ans == 'Y'):
    ans = input("Задать параметры по номеру варианта? ")
    if ans == 'y' or ans == 'Y':
        nvar = int(input("Номер варианта: "))
        fl = nvar
        fh = 10 * (nvar + 1)
        f0 = 0.5 * (fh + fl)
        Z0 = 10 * nvar
        print("Нижняя граничная частота: {}".format(fl))
        print("Верхняя граничная частота: {}".format(fh))
        print("Волновое сопротивление линии на частоте {0: 5.2f} Z0: {1: .1f}".format(f0, Z0))
    else:
        fl = float(input("Нижняя граничная частота: "))
        fh = float(input("Верхняя граничная частота: "))
        f0 = 0.5 * (fh + fl)
        Z0 = float(input("Волновое сопротивление линии на частоте {0: 5.2f}: ".format(f0)))

    nsig = int(input(
        "\nСигнал возбуждения ЛП:\n0 - ВЧ сигнал,\n1 - ВЧ импульс,\n2 - НЧ импульс,\n3 - НЧ+ВЧ импульс,\n4 - ШП импульс\n"))
    if 0 <= nsig <= 1 or nsig == 3:
        fc = float(input("Частота сигнала: "))
    else:
        fc = f0
    G = float(input("Проводимость G "))
    K0 = float(input("Согласование слева  K0 (R0 = K0 * Z0): "))
    KN = float(input("Согласование справа KN (RN = KN * Z0): "))
    Time = float(input("Временной интервал: "))
    Nc = int(input("Число ячеек в ЛП: "))
    A0 = 1.0
    AN = 0.0
else:
    ################################################
    #####   Параметры, заданные по умолчанию   #####
    ################################################
    nvar = 30  # Задайте здесь свой номер варианта!
    nsig = 0
    fl = nvar
    fh = 10 * (nvar + 1)
    f0 = 0.5 * (fh + fl)
    Z0 = 10 * nvar
    fc = f0  # 0.95*fh
    G = 0
    K0 = 1.0
    KN = 1.0
    A0 = 1.0
    AN = 0.0
    Time = 100
    Nc = 10
################################################
lffilter = False
fwFront = bwFront = Time * 0.005
Start = Time * 0.05 if nsig in [1, 2, 3] else 0
Stop = Time * 0.1 if nsig in [1, 2, 3] else Time
time0 = Time * 0.075 if nsig in [1, 2, 3] else Time * 0.5

print("\n{:*^60}".format(""))
print("{:*^60}".format(" Исходные данные для расчета: "))
print("{:*^60}".format(""))
if nsig == 0:
    print("Сигнал возбуждения ЛП: ВЧ сигнал")
elif nsig == 1:
    print("Сигнал возбуждения ЛП: ВЧ импульс")
elif nsig == 2:
    print("Сигнал возбуждения ЛП: НЧ импульс")
elif nsig == 3:
    print("Сигнал возбуждения ЛП: НЧ+ВЧ импульс")
elif nsig == 4:
    print("Сигнал возбуждения ЛП: ШП импульс")
else:
    print("Сигнал задан неверно!")
    exit(-1)

print("Граничные частоты: fl = {0: f}, fh = {1: f}".format(fl, fh))
print("Волновое сопротивление на частоте {0: 5.2f}: Z0 = {1: .1f}".format(f0, Z0))
print("Частота сигнала: {}".format(fc))
print("Проводимость G: {}".format(G))
print("Согласование слева K0: {} ".format(K0))
print("Согласование справа KN: {}".format(KN))
print("Временной интервал: {}".format(Time))
print("Число ячеек в ЛП: {}".format(Nc))
print("{:*^60}\n".format(""))

scale = 10 ** int(log10(fh))  # Масштабирующий коэффициент
fl /= scale
fh /= scale
f0 /= scale  # 0.5*(fl+fh)
fc /= scale

# Расчет параметров эквивалентной схемы отдельной ячейки ЛП
L = sqrt(Z0 ** 2 * f2w(f0) ** 2 * (2.0 * f2w(fh) ** 2 - f2w(fl) ** 2 - f2w(f0) ** 2) / (
            (f2w(fh) ** 2 - f2w(fl) ** 2) ** 2 * (f2w(f0) ** 2 - f2w(fl) ** 2)))
C1 = 1.0 / (L * (f2w(fh) ** 2 - f2w(fl) ** 2))
C2 = 1.0 / (f2w(fl) ** 2 * L)

dpp = 20 if nsig in [2, 3] else 40  # Количество точек на период гармонического сигнала
npp = 50  # Количество итераций для решения уравнений возбуждения
dt = 1.0 / (fc * dpp)  # Шаг интегрирования уравнений возбуждения линии передачи
num = int(Time / dt + 1)  # Количество расчетных точек на временном интервале

print("{:*^60}".format(""))
print("Параметры эквивалентной схемы:")
print("L    = {0: f},\nC1   = {1: f},\nC2   = {2: f},\nG/C2 = {3: f}".format(L, C1, C2, G / C2))
print("{:*^60}\n".format(""))

# Расчет зависимостей фазового сдвига на ячейку и волнового сопротивления ЛП от частоты
# fq = np.linspace(0.8*fl, fh*1.2, num)
fq = np.linspace(0.1 * fl, fh * 1.2, num)

Gama = Gam(fq, L, C1, C2, G)
Zw = Zw(fq, L, C1, C2, G)
dF = (Gam(fq + 0.1, L, C1, C2, G).imag - Gam(fq - 0.1, L, C1, C2, G).imag) / 0.2

if show_tmp_graph: lineplot(fq, Gama.imag, '#539caf', "Частота, отн.ед.", "Фазовый сдвиг на ячейку, рад.",
                            "Фазовая характеристика")
if show_tmp_graph: lineplot(fq, Zw.real, '#539caf', "Частота, отн.ед.", "Z, Ом", "Волновое сопротивление")

aU = [0] * Nc  # Массив напряжений на емкости C2
dU = [0] * Nc  # Массив производных напряжений на емкости C2
aV = [0] * (Nc + 1)  # Массив напряжений на емкости C1
dV = [0] * (Nc + 1)  # Массив производных напряжений на емкости C1
Vinp = [0] * num  # Массив входных напряжений
Vout = [0] * num  # Массив выходных напряжений
time = [0] * num  # Массив временных отсчетов
freq = [0] * num  # Массив частотных гармоник

# Коэффициенты разложения в ряд Фурье для входного и выходного сигналов
a1 = [0] * num
b1 = [0] * num
a2 = [0] * num
b2 = [0] * num

# Массивы для построения интерполированных распределений напряжений вдоль ЛП
Vz = [0] * dpp
tz = [0] * dpp
Nz = [0] * (Nc + 1)

for i in range(dpp):
    Vz[i] = [0] * (Nc + 1)
hpp = Time / (dpp - 1) if nsig in [2, 3] else dt
ipp = 0

if save_csv_files:
    fp1 = open('inp_out.csv', 'w', encoding='utf-8')
    fp2 = open('cells.csv', 'w', encoding='utf-8')
    fp3 = open('spectrum.csv', 'w', encoding='utf-8')
    fp4 = open('data.csv', 'w', encoding='utf-8')

for i in range(Nc + 1):
    if save_csv_files: fp2.write("; {0: d}".format(i))
    Nz[i] = i;
if save_csv_files: fp2.write("\n")

# Основной цикл по времени. Здесь решается система уравнений возбуждения ЛП.
for it in range(num):
    time[it] = dt * it
    # Решение уравнений возбуждения линии передачи
    for i in range(npp):
        dV[0] += (1.0 / (L * C1) * (aV[1] - aV[0] + aU[0]) + 1.0 / (Z0 * K0 * C1) * (
                    A0 * d_signal(dt * it) - dV[0])) * dt / npp
        for ic in range(Nc):
            if not lffilter: dU[ic] += (1.0 / (L * C2) * (aV[ic] - aV[ic + 1] - aU[ic]) - G / C2 * dU[ic]) * dt / npp
            if ic == 0:
                continue
            dV[ic] += 0.5 / (L * C1) * (aV[ic - 1] - 2.0 * aV[ic] + aV[ic + 1] + aU[ic] - aU[ic - 1]) * dt / npp
        dV[Nc] += (1.0 / (L * C1) * (aV[Nc - 1] - aV[Nc] - aU[Nc - 1]) + 1.0 / (Z0 * KN * C1) * (
                    AN * d_signal(dt * it) - dV[Nc])) * dt / npp
        for ic in range(Nc):
            aV[ic] += dV[ic] * dt / npp
            aU[ic] += dU[ic] * dt / npp
        aV[Nc] += dV[Nc] * dt / npp

    if save_csv_files: fp2.write("{0: f}".format(time[it]))
    for ic in range(Nc):
        if save_csv_files: fp2.write("; {0: f}".format(aV[ic]))
        if (ipp * hpp + time0 <= dt * it) and ipp < dpp:
            Vz[ipp][ic] = aV[ic]
    if save_csv_files: fp2.write("; {0: f}\n".format(aV[Nc]))
    if (ipp * hpp + time0 <= dt * it) and ipp < dpp:
        Vz[ipp][Nc] = aV[Nc]
        tz[ipp] = time[it]
        ipp = ipp + 1
    Vinp[it] = aV[0]  # A0*signal(it*dt) - aV[0]
    Vout[it] = aV[Nc]  # aV[Nc] - AN*signal(it*dt)
    if it % 100 == 0:
        print('{0: 7.3f} {1: 7.3f} {2: 7.3f} '.format(time[it], Vinp[it], Vout[it]))
    if save_csv_files: print('{0}; {1}; {2}; '.format(time[it], Vinp[it], Vout[it]), file=fp1)

if save_csv_files:
    fp1.close()
    fp2.close()

if show_tmp_graph: lineplot(time, Vinp, '#539caf', "Время, отн.ед.", "Напряжение, В", "Входной сигнал")
if show_tmp_graph: lineplot(time, Vout, '#539caf', "Время, отн.ед.", "Напряжение, В", "Выходной сигнал")

print('Расчет спектра входного сигнала...')
sp_inp = np.fft.fft(Vinp)
print('Расчет спектра выходного сигнала...')
sp_out = np.fft.fft(Vout)
trans = (sp_out.real / sp_inp.real)

if save_csv_files: fp3.write("Frequency; V input; V output\n")
for i in range(num):
    freq[i] = i / Time
    a1[i] = hypot(sp_inp[i].real, sp_inp[i].imag) / num * 2
    a2[i] = hypot(sp_out[i].real, sp_out[i].imag) / num * 2
    b1[i] = a2[i] / a1[i]
    if save_csv_files: print('{0}; {1}; {2};'.format(freq[i], a1[i], a2[i]), file=fp3)
if show_tmp_graph:
    lineplot(freq[0:num // 2], a1[0:num // 2], '#539caf', "Частота, отн.ед.", "Напряжение, В",
             "Спектр входного сигнала")
    lineplot(freq[0:num // 2], a2[0:num // 2], '#539caf', "Частота, отн.ед.", "Напряжение, В",
             "Спектр выходного сигнала")
    if nsig == 4: lineplot(freq[0:num // 2], b1[0:num // 2], '#539caf', "Частота, отн.ед.", "Коэффициент передачи",
                           "Амплитудно-частотная характеристика")
if save_csv_files: fp3.close()

if save_csv_files:
    fp4.write(";".format(i))
    for j in range(Nc + 1): fp4.write("{0: d}; ".format(j))
    fp4.write("\n".format(i))
    for i in range(dpp):
        fp4.write("{0: f}; ".format(tz[i]))
        for j in range(Nc + 1):
            fp4.write("{0: f}; ".format(Vz[i][j]))
        fp4.write("\n")
    fp4.close()

# Построение графиков
_, ax = plt.subplots(nrows=3, ncols=2)
ax1, ax2, ax3, ax4, ax5, ax6 = ax.flatten()
gridsize = (3, 2)
ax1 = plt.subplot2grid(gridsize, (0, 0))
ax2 = plt.subplot2grid(gridsize, (0, 1))
ax3 = plt.subplot2grid(gridsize, (1, 0))
ax4 = plt.subplot2grid(gridsize, (1, 1))
ax5 = plt.subplot2grid(gridsize, (2, 0))
ax6 = plt.subplot2grid(gridsize, (2, 1))

ax1.plot(time, Vinp, lw=1, color='#539caf', alpha=1)
ax1.set_xlim(time[0], time[-1])
ax1.set_ylim(min(Vinp) * 1.1, max(Vinp) * 1.1)
ax1.set_title("(а) - $U_{вх}(t)$", fontsize=10)
ax1.set_xlabel(r"Время $\times$ $10^{" + "-{0:d}".format(int(log10(scale))) + "}$, сек.", fontsize=10)
ax1.set_ylabel("$U_{вх}$, В", fontsize=10)

ax2.plot(time, Vout, lw=1, color='#539caf', alpha=1)
ax2.set_xlim(time[0], time[-1])
ax2.set_ylim(min(Vout) * 1.1, max(Vout) * 1.1)
ax2.set_title("(б) - $U_{вых}(t)$", fontsize=10)
ax2.set_xlabel(r"Время $\times$ $10^{" + "-{0:d}".format(int(log10(scale))) + "}$, сек.", fontsize=10)
ax2.set_ylabel("$U_{вых}$, В", fontsize=10)

ax3.plot(freq[0:num // 2], a1[0:num // 2], lw=1, color='#539caf', alpha=1)
ax3.set_xlim(0, max(fh, fc) * 1.05)
ax3.set_ylim(min(a1[0:num // 2]) * 1.1, max(a1[0:num // 2]) * 1.05)
ax3.set_title("(в) - $U_{вх}(f)$", fontsize=10)
ax3.set_xlabel(r"Частота $\times$ $10^{" + "{0:d}".format(int(log10(scale))) + "}$, Гц", fontsize=10)
ax3.set_ylabel("$U_{вх}$, В", fontsize=10)

ax4.plot(freq[0:num // 2], a2[0:num // 2], lw=1, color='#539caf', alpha=1)
ax4.set_xlim(0, max(fh, fc) * 1.05)
ax4.set_ylim(min(a2[0:num // 2]) * 1.1, max(a2[0:num // 2]) * 1.05)
ax4.set_title("(г) - $U_{вых}(f)$", fontsize=10)
ax4.set_xlabel(r"Частота $\times$ $10^{" + "{0:d}".format(int(log10(scale))) + "}$, Гц", fontsize=10)
ax4.set_ylabel("$U_{вых}$, В", fontsize=10)

if nsig == 4:
    ax5.plot(freq[0:num // 2], b1[0:num // 2], lw=1, alpha=1)
    ax5.plot(fq, np.exp(-Gama.real), lw=1, alpha=1)
    ax5.set_xlim(0, max(fh, fc) * 1.5)  # 1.1
    ax5.set_title("(д) - $K(f)$", fontsize=10)
    ax5.set_xlabel(r"Частота $\times$ $10^{" + "{0:d}".format(int(log10(scale))) + "}$, Гц", fontsize=10)
    ax5.set_ylabel(r"K=$\frac{U_{вых}}{U_{вх}}$", fontsize=10)
else:
    color = 'tab:blue'
    ax5.plot(fq, Gama.imag, lw=1, color=color, alpha=1)
    ax5.set_xlim(0, max(fh, fc) * 1.1)
    ax5.set_ylim(min(Gama.imag), max(Gama.imag))
    ax5.set_title(r"(д) - $Z_0(f)$, $\phi(f)$", fontsize=10)
    ax5.set_xlabel(r"Частота $\times$ $10^{" + "{0:d}".format(int(log10(scale))) + "}$, Гц", fontsize=10)
    ax5.set_ylabel(r"$\phi$, рад.", color=color, fontsize=10)
    ax5.tick_params(axis='y', labelcolor=color)

    ax5 = ax5.twinx()
    color = 'tab:red'
    ax5.plot(fq, Zw.real, lw=1, color=color, alpha=1)
    ax5.set_ylim(min(Zw.real), max(Zw.real))
    ax5.set_ylabel("$Z_0$, Ом", color=color, fontsize=10)
    ax5.tick_params(axis='y', labelcolor=color)

ax6.set_prop_cycle(color=['r', 'g', 'b', 'y', 'c'], linestyle=['-', '--', ':', '-.', '-'], lw=5 * [0.5])
ax6.set_alpha(0.5)
Vzs = np.array(Nc + 1)
Ncs = np.linspace(0, Nc, Nc + 1)
Nzsp = np.linspace(0, Nc, (Nc + 1) * dpp)
for i in range(dpp):
    Vzs = Vz[i]
    spl = make_interp_spline(Ncs, Vzs, k=3)
    ax6.plot(Nzsp, spl(Nzsp))
ax6.set_xlim(0, Nc)
ax6.set_title("(е) - $U_i$", fontsize=10)
ax6.set_xlabel("Номер ячейки $(i)$", fontsize=10)
ax6.set_ylabel("$U_i$, В", fontsize=10)

_.tight_layout()
plt.show()