import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

# Время
T = np.linspace(0, 10, 1000)

# Символьная переменная
t = sp.Symbol('t')

# Функции r(t) и phi(t)
r = 5 - 0.5 * t
phi = 2 * t

# Координаты в декартовой системе
x = r * sp.cos(phi)
y = r * sp.sin(phi)

# Производные
x_diff = sp.diff(x, t)
y_diff = sp.diff(y, t)

x_diff2 = sp.diff(x_diff, t)
y_diff2 = sp.diff(y_diff, t)

# Массивы для хранения значений
X = np.zeros_like(T)
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
AX = np.zeros_like(T)
AY = np.zeros_like(T)

# Вычисление значений
for i in range(len(T)):
    X[i] = float(x.subs(t, T[i]))
    Y[i] = float(y.subs(t, T[i]))
    VX[i] = float(x_diff.subs(t, T[i]))
    VY[i] = float(y_diff.subs(t, T[i]))
    AX[i] = float(x_diff2.subs(t, T[i]))
    AY[i] = float(y_diff2.subs(t, T[i]))

# Модули скорости и ускорения
v_magnitudes = np.sqrt(VX**2 + VY**2)
a_magnitudes = np.sqrt(AX**2 + AY**2)

max_v = np.max(v_magnitudes)
max_a = np.max(a_magnitudes)

# Создание графика
fig, ax = plt.subplots()
ax.axis('equal')
a_lim = 0.8
ax.set_xlim([min(X) - a_lim, max(X) + a_lim])
ax.set_ylim([min(Y) - a_lim, max(Y) + a_lim])

point, = ax.plot([], [], 'go', markersize=10)
ax.plot(X, Y, 'r-', lw=1)
velocity_line, = ax.plot([], [], 'b-', lw=1)
velocity_arrow_head, = ax.plot([], [], 'b-')
acceleration_line, = ax.plot([], [], 'g-', lw=1)
acceleration_arrow_head, = ax.plot([], [], 'g-')
radius_vector_line, = ax.plot([], [], 'y-', lw=1)
radius_vector_arrow_head, = ax.plot([], [], 'y-')
curvature_radius_line, = ax.plot([], [], 'm--', lw=1)
curvature_radius_arrow_head, = ax.plot([], [], 'm--')

def rotate_2d(x_arr, y_arr, angle):
    x_new = x_arr * np.cos(angle) - y_arr * np.sin(angle)
    y_new = x_arr * np.sin(angle) + y_arr * np.cos(angle)
    return x_new, y_new

def update(frame):
    x0 = X[frame]
    y0 = Y[frame]
    vx = VX[frame]
    vy = VY[frame]
    ax0 = AX[frame]
    ay0 = AY[frame]

    point.set_data([x0], [y0])

    v_mag = math.sqrt(vx**2 + vy**2)
    a_mag = math.sqrt(ax0**2 + ay0**2)

    v_scale = v_mag / max_v if max_v != 0 else 0
    a_scale = a_mag / max_a if max_a != 0 else 0

    if v_mag != 0:
        vx_norm = vx / v_mag
        vy_norm = vy / v_mag
    else:
        vx_norm, vy_norm = 0, 0

    if a_mag != 0:
        ax_norm = ax0 / a_mag
        ay_norm = ay0 / a_mag
    else:
        ax_norm, ay_norm = 0, 0

    vx_draw = vx_norm * v_scale
    vy_draw = vy_norm * v_scale
    ax_draw = ax_norm * a_scale
    ay_draw = ay_norm * a_scale

    velocity_line.set_data([x0, x0 + vx_draw], [y0, y0 + vy_draw])
    angle_v = math.atan2(vy_draw, vx_draw)
    arrow_x = np.array([-0.08, 0, -0.08])
    arrow_y = np.array([0.04, 0, -0.04])
    VArrowX, VArrowY = rotate_2d(arrow_x, arrow_y, angle_v)
    velocity_arrow_head.set_data(
        VArrowX + x0 + vx_draw, VArrowY + y0 + vy_draw)

    acceleration_line.set_data([x0, x0 + ax_draw], [y0, y0 + ay_draw])
    angle_a = math.atan2(ay_draw, ax_draw)
    AArrowX, AArrowY = rotate_2d(arrow_x, arrow_y, angle_a)
    acceleration_arrow_head.set_data(
        AArrowX + x0 + ax_draw, AArrowY + y0 + ay_draw)

    # Рисуем радиус-вектор
    radius_vector_line.set_data([0, x0], [0, y0])
    angle_r = math.atan2(y0, x0)
    RArrowX, RArrowY = rotate_2d(arrow_x, arrow_y, angle_r)
    radius_vector_arrow_head.set_data(RArrowX + x0, RArrowY + y0)

    numerator = (vx**2 + vy**2)**1.5
    denominator = abs(vx * ay0 - vy * ax0)
    if denominator != 0:
        R_curv = numerator / denominator
    else:
        R_curv = np.inf

    norm_vx = -vy
    norm_vy = vx
    norm = np.hypot(norm_vx, norm_vy)
    if norm != 0:
        norm_vx /= norm
        norm_vy /= norm

    center_x = x0 + R_curv * norm_vx
    center_y = y0 + R_curv * norm_vy

    curvature_radius_line.set_data([x0, center_x], [y0, center_y])
    angle_c = math.atan2(center_y - y0, center_x - x0)
    CArrowX, CArrowY = rotate_2d(arrow_x, arrow_y, angle_c)
    curvature_radius_arrow_head.set_data(
        CArrowX + center_x, CArrowY + center_y)

    return (point,
            velocity_line, velocity_arrow_head,
            acceleration_line, acceleration_arrow_head,
            radius_vector_line, radius_vector_arrow_head,
            curvature_radius_line, curvature_radius_arrow_head)

ani = animation.FuncAnimation(
    fig, update, frames=len(T), interval=20, blit=True)
plt.show()