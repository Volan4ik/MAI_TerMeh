import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from scipy.integrate import odeint
import matplotlib.gridspec as gridspec

# Функция для поворота координат на угол Alpha
def Rot2D(X, Y, Alpha):
    RX = X * np.cos(Alpha) - Y * np.sin(Alpha)
    RY = X * np.sin(Alpha) + Y * np.cos(Alpha)
    return RX, RY

# Параметры системы
m1 = 10.0
m2 = 1.0
R = 1.0
l = 0.5
M0 = 0.0
gamma = 3 * math.pi / 2.0
k = 10.0
g = 9.81

# Начальные условия
t0 = 0.0
phi0 = 0.0
psi0 = math.pi / 6
dphi0 = 0.0
dpsi0 = 0.0
y0 = [phi0, psi0, dphi0, dpsi0]

# Дискретизация времени
Tmax = 45.0
Nsteps = 1000
T = np.linspace(t0, Tmax, Nsteps)

# Система дифференциальных уравнений
def SystDiffEq(y, t, m1, m2, R, l, M0, gamma, k, g):
    phi, psi, dphi, dpsi = y

    sqrt_Rl = math.sqrt(R**2 - l**2)
    sin_psi_phi = math.sin(psi - phi)
    cos_psi_phi = math.cos(psi - phi)

    # Коэффициенты матрицы инерции
    A11 = (2 * m1 + m2) * R**2
    A12 = m2 * R * sqrt_Rl * cos_psi_phi
    A21 = sqrt_Rl * R * cos_psi_phi
    A22 = R**2 - (2.0 / 3.0) * l**2

    # Внешние силы и моменты
    B1 = M0 * math.sin(gamma * t) - k * dphi - (m1 + m2) * g * R * \
        math.sin(phi) + m2 * R * sqrt_Rl * (dpsi**2) * sin_psi_phi
    B2 = -sqrt_Rl * (R * (dphi**2) * sin_psi_phi + g * math.sin(psi))

    # Решение системы линейных уравнений
    detA = A11 * A22 - A12 * A21
    detA1 = B1 * A22 - A12 * B2
    detA2 = A11 * B2 - B1 * A21

    ddphi = detA1 / detA  # Угловое ускорение первого тела
    ddpsi = detA2 / detA  # Угловое ускорение второго тела

    return [dphi, dpsi, ddphi, ddpsi]

# Решение системы дифференциальных уравнений
Y = odeint(SystDiffEq, y0, T, args=(m1, m2, R, l, M0, gamma, k, g))
phi_array = Y[:, 0]  # Углы phi для всех моментов времени
psi_array = Y[:, 1]  # Углы psi для всех моментов времени
dphi_array = Y[:, 2]  # Угловые скорости dphi для всех моментов времени
dpsi_array = Y[:, 3]  # Угловые скорости dpsi для всех моментов времени

# Вычисление ускорений
ddphi_array = np.zeros(Nsteps)
ddpsi_array = np.zeros(Nsteps)
for i in range(Nsteps):
    y_i = Y[i]
    t_i = T[i]
    derivs = SystDiffEq(y_i, t_i, m1, m2, R, l, M0, gamma, k, g)
    ddphi_array[i] = derivs[2]
    ddpsi_array[i] = derivs[3]

# Вычисление сил реакции
Nx = - (m1 + m2) * R * (ddphi_array * np.cos(phi_array) - dphi_array**2 * np.sin(phi_array)) \
     - m2 * (R**2 - l**2) * (ddpsi_array * np.cos(psi_array) -
                             dpsi_array**2 * np.sin(psi_array))

Ny = - (m1 + m2) * R * (ddphi_array * np.sin(phi_array) + dphi_array**2 * np.cos(phi_array)) \
     - (m1 + m2) * g \
     - m2 * (R**2 - l**2) * (ddpsi_array * np.sin(psi_array) +
                             dpsi_array**2 * np.cos(psi_array))

# Вычисление координат точек
x_O = -R * np.sin(phi_array)
y_O = R * np.cos(phi_array)

r = math.sqrt(R**2 - l**2)

x_C = x_O - r * np.sin(psi_array)
y_C = y_O + r * np.cos(psi_array)

x_rel = -r * np.sin(psi_array)
y_rel = r * np.cos(psi_array)

# Инвертирование координат для удобства отображения
x_O_rot = -x_O
y_O_rot = -y_O
x_C_rot = -x_C
y_C_rot = -y_C
x_rel_rot = -x_rel
y_rel_rot = -y_rel

# Настройка графиков и анимации
fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 2], wspace=0.3)

# График анимации
ax_anim = plt.subplot(gs[0])
ax_anim.set_xlim(-2.5, 2.5)
ax_anim.set_ylim(-3, 3)
ax_anim.set_xlabel('ось x')
ax_anim.set_ylabel('ось y')
ax_anim.set_aspect('equal')
ax_anim.set_title('Анимация системы')

# Начальное состояние графиков
PointO1, = ax_anim.plot([0], [0], 'bo')
Circ_Angle = np.linspace(0, 2 * math.pi, 100)
Circ, = ax_anim.plot(x_O_rot[0] + R * np.cos(Circ_Angle),
                     y_O_rot[0] + R * np.sin(Circ_Angle), 'g')

# Начальное состояние стержня
ArrowX = np.array([0, 0, 0])
ArrowY = np.array([-l, 0, l])
initial_angle = math.atan2(y_rel_rot[0], x_rel_rot[0])
R_Stick_ArrowX, R_Stick_ArrowY = Rot2D(ArrowX, ArrowY, initial_angle)
Stick_Arrow, = ax_anim.plot(
    R_Stick_ArrowX + x_C_rot[0], R_Stick_ArrowY + y_C_rot[0], 'k-')

# Линии, соединяющие точки
O1O, = ax_anim.plot([0, x_O_rot[0]], [0, y_O_rot[0]], 'b:')
OC, = ax_anim.plot([x_O_rot[0], x_C_rot[0]], [
                   y_O_rot[0], y_C_rot[0]], 'b-')

# Настройка графиков зависимостей
gs_plots = gridspec.GridSpecFromSubplotSpec(
    2, 2, subplot_spec=gs[1], wspace=0.4, hspace=0.6)

# График зависимости phi(t)
ax_phi = plt.subplot(gs_plots[0, 0])
ax_phi.set_xlim(t0, 10)
ax_phi.set_ylim(min(phi_array) * 1.1, max(phi_array) * 1.1)
ax_phi.set_xlabel('Время (с)')
ax_phi.set_ylabel('phi(t) (рад)')
line_phi, = ax_phi.plot([], [], 'r-')
ax_phi.grid(True)

# График зависимости psi(t)
ax_psi = plt.subplot(gs_plots[0, 1])
ax_psi.set_xlim(t0, 10)
ax_psi.set_ylim(min(psi_array) * 1.1, max(psi_array) * 1.1)
ax_psi.set_xlabel('Время (с)')
ax_psi.set_ylabel('psi(t) (рад)')
line_psi, = ax_psi.plot([], [], 'b-')
ax_psi.grid(True)

# График зависимости Nx(t)
ax_Nx = plt.subplot(gs_plots[1, 0])
ax_Nx.set_xlim(t0, 10)
ax_Nx.set_ylim(min(Nx) * 1.1, max(Nx) * 1.1)
ax_Nx.set_xlabel('Время (с)')
ax_Nx.set_ylabel('Nx (Н)')
line_Nx, = ax_Nx.plot([], [], 'y-')
ax_Nx.grid(True)

# График зависимости Ny(t)
ax_Ny = plt.subplot(gs_plots[1, 1])
ax_Ny.set_xlim(t0, 10)
padding = 0.1 * max(abs(min(Ny)), abs(max(Ny)))
ax_Ny.set_ylim(min(Ny) - padding, max(Ny) + padding)
ax_Ny.set_xlabel('Время (с)')
ax_Ny.set_ylabel('Ny (Н)')
line_Ny, = ax_Ny.plot([], [], 'g-')
ax_Ny.grid(True)

# Данные для графиков
phi_xdata, phi_ydata = [], []
psi_xdata, psi_ydata = [], []
Nx_xdata, Nx_ydata = [], []
Ny_xdata, Ny_ydata = [], []

# Функция для обновления анимации
def anima(i):
    # Обновление линий и окружности
    O1O.set_data([0, x_O_rot[i]], [0, y_O_rot[i]])  # Линия от начала координат до точки O
    OC.set_data([x_O_rot[i], x_C_rot[i]], [y_O_rot[i], y_C_rot[i]])  # Линия от точки O до точки C
    Circ.set_data(x_O_rot[i] + R * np.cos(Circ_Angle),
                  y_O_rot[i] + R * np.sin(Circ_Angle))  # Окружность вокруг точки O
    current_angle = math.atan2(y_rel_rot[i], x_rel_rot[i])  # Угол поворота стержня
    R_Stick_ArrowX, R_Stick_ArrowY = Rot2D(ArrowX, ArrowY, current_angle)  # Поворот стержня
    Stick_Arrow.set_data(
        R_Stick_ArrowX + x_C_rot[i], R_Stick_ArrowY + y_C_rot[i])  # Обновление стержня

    # Обновление графиков зависимостей
    phi_xdata.append(T[i])
    phi_ydata.append(phi_array[i])
    line_phi.set_data(phi_xdata, phi_ydata)

    psi_xdata.append(T[i])
    psi_ydata.append(psi_array[i])
    line_psi.set_data(psi_xdata, psi_ydata)

    Nx_xdata.append(T[i])
    Nx_ydata.append(Nx[i])
    line_Nx.set_data(Nx_xdata, Nx_ydata)

    Ny_xdata.append(T[i])
    Ny_ydata.append(Ny[i])
    line_Ny.set_data(Ny_xdata, Ny_ydata)

    # Сдвиг окна отображения
    window = 10
    if T[i] > window:
        ax_phi.set_xlim(T[i] - window, T[i])
        ax_psi.set_xlim(T[i] - window, T[i])
        ax_Nx.set_xlim(T[i] - window, T[i])
        ax_Ny.set_xlim(T[i] - window, T[i])

        # Удаление старых данных
        while phi_xdata and phi_xdata[0] < T[i] - window:
            phi_xdata.pop(0)
            phi_ydata.pop(0)
        line_phi.set_data(phi_xdata, phi_ydata)

        while psi_xdata and psi_xdata[0] < T[i] - window:
            psi_xdata.pop(0)
            psi_ydata.pop(0)
        line_psi.set_data(psi_xdata, psi_ydata)

        while Nx_xdata and Nx_xdata[0] < T[i] - window:
            Nx_xdata.pop(0)
            Nx_ydata.pop(0)
        line_Nx.set_data(Nx_xdata, Nx_ydata)

        while Ny_xdata and Ny_xdata[0] < T[i] - window:
            Ny_xdata.pop(0)
            Ny_ydata.pop(0)
        line_Ny.set_data(Ny_xdata, Ny_ydata)

    else:
        ax_phi.set_xlim(t0, window)
        ax_psi.set_xlim(t0, window)
        ax_Nx.set_xlim(t0, window)
        ax_Ny.set_xlim(t0, window)

    return O1O, OC, Circ, Stick_Arrow, line_phi, line_psi, line_Nx, line_Ny

# Создание анимации
anim = FuncAnimation(fig, anima, frames=Nsteps, interval=20, blit=False)

# Отображение анимации
plt.tight_layout()
plt.show()