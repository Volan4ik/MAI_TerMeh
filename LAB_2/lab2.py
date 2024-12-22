import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math

# Функция для поворота координат на угол Alpha
def Rot2D(X, Y, Alpha):
    RX = X * np.cos(Alpha) - Y * np.sin(Alpha)
    RY = X * np.sin(Alpha) + Y * np.cos(Alpha)
    return RX, RY

# Параметры системы
R = 5
l = 2
r = math.sqrt(R * R - l * l)

# Символьная переменная времени
t = sp.Symbol('t')

# Углы поворота, зависящие от времени
phi = 0.3 * t
ksi = 0.5 * t

# Координаты точки O (движение по окружности радиуса R)
x_O = -R * sp.sin(phi)
y_O = R * sp.cos(phi)

# Координаты точки C (относительно точки O)
x_C = x_O - r * sp.sin(ksi)
y_C = y_O + r * sp.cos(ksi)

# Относительные координаты точки C относительно точки O
x_rel = -r * sp.sin(ksi)
y_rel = r * sp.cos(ksi)

# Скорость точки C (производная координат по времени)
Vx_C = sp.diff(x_C, t)
Vy_C = sp.diff(y_C, t)
V_mod_C = sp.sqrt(Vx_C**2 + Vy_C**2)

# Ускорение точки C (вторая производная координат по времени)
Ax_C = sp.diff(x_C, t, 2)
Ay_C = sp.diff(y_C, t, 2)
A_mod_C = sp.sqrt(Ax_C**2 + Ay_C**2)

# Массив времени для анимации
T = np.linspace(0, 45, 1000)

# Преобразование символьных выражений в численные функции
X_O_def = sp.lambdify(t, x_O, modules='numpy')
Y_O_def = sp.lambdify(t, y_O, modules='numpy')
X_C_def = sp.lambdify(t, x_C, modules='numpy')
Y_C_def = sp.lambdify(t, y_C, modules='numpy')
X_REL_def = sp.lambdify(t, x_rel, modules='numpy')
Y_REL_def = sp.lambdify(t, y_rel, modules='numpy')

# Вычисление координат для всех моментов времени
X_O = X_O_def(T)
Y_O = Y_O_def(T)
X_C = X_C_def(T)
Y_C = Y_C_def(T)
X_REL = X_REL_def(T)
Y_REL = Y_REL_def(T)

# Настройка графика
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_aspect('equal')
ax1.set_xlim([-10, 10])
ax1.set_ylim([-20, 20])
ax1.set_xlabel('Ось x')
ax1.set_ylabel('Ось y')
ax1.invert_xaxis()
ax1.invert_yaxis()
ax1.grid(True)

# Начальное состояние графиков
PointO1, = ax1.plot([0], [0], 'bo')
Circ_Angle = np.linspace(0, 2 * np.pi, 100)
Circ, = ax1.plot(X_O[0] + R * np.cos(Circ_Angle), Y_O[0] + R * np.sin(Circ_Angle), 'g')

# Начальное состояние стержня
ArrowX = np.array([0, 0, 0])
ArrowY = np.array([l, 0, -l])
R_Stick_ArrowX, R_Stick_ArrowY = Rot2D(ArrowX, ArrowY, math.atan2(Y_REL[0], X_REL[0]))
Stick_Arrow, = ax1.plot(R_Stick_ArrowX + X_C[0], R_Stick_ArrowY + Y_C[0], 'r-')

# Линии, соединяющие точки
O1O, = ax1.plot([0, X_O[0]], [0, Y_O[0]], 'b:')
OC, = ax1.plot([X_O[0], X_C[0]], [Y_O[0], Y_C[0]], 'b:')

# Функция для обновления анимации
def anima(i):
    O1O.set_data([0, X_O[i]], [0, Y_O[i]])
    OC.set_data([X_O[i], X_C[i]], [Y_O[i], Y_C[i]])
    Circ.set_data(X_O[i] + R * np.cos(Circ_Angle), Y_O[i] + R * np.sin(Circ_Angle))

    # Поворот стержня
    angle = math.atan2(Y_REL[i], X_REL[i])
    R_Stick_ArrowX, R_Stick_ArrowY = Rot2D(ArrowX, ArrowY, angle)
    Stick_Arrow.set_data(R_Stick_ArrowX + X_C[i], R_Stick_ArrowY + Y_C[i])

    return O1O, OC, Circ, Stick_Arrow, PointO1

anim = FuncAnimation(fig, anima, frames=len(T), interval=20, blit=True)

plt.show()