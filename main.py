#from mpl_toolkits.mplot3d import *
#from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import sympy as sy
import math
from trilateration import sphereCircle, Trilateration_3D
from matplotlib.animation import FuncAnimation
from IPython.display import Image
from IPython.display import display, clear_output
from typing import Any
from dataclasses import dataclass


def ggg():
    pass

# How many towers. All towers receive the transmission.
num_towers = 4

# Metre length of a square containing the transmitting
# device, centred around (x, y) = (0, 0). Device will be randomly placed
# in this area.
tx_square_side = 5e3

# Metre length of a square containing the towers,
# centred around (x, y) = (0, 0). towers will be randomly placed
# in this area.
rx_square_side = 30e3

# Speed of transmission propogation. Generally equal to speed of
# light for radio signals.
v = 299792458

# Time at which transmission is performed. Really just useful to
# make sure the code is using relative times rather than depending on one
# of the receive times being zero.
t_0 = 2.5

# Metre increments to radii of circles when generating locus of
# circle intersection.
delta_d = int(100)

# Max distance a transmission will be from the tower that first
# received the transmission. This puts an upper bound on the radii of the
# circle, thus limiting the size of the locus to be near the towers.
max_d = int(20e3)

# Standard deviation of noise added to the
# receive times at the towers. Mean is zero.
rec_time_noise_stdd = 1e-12

# Whether to plot circles that would be
# used if performing trilateration. These are circles that are centred
# on the towers and touch the transmitter site.
plot_trilateration_tx = True
plot_trilateration_spheres = True
plot_trilateration_spheresIntersection_circles = True

# Whether to plot a straight line
# between every pair of towers. This is useful for visualising the
# hyperbolic loci focal points.
plot_lines_between_towers = False
plot_lines_to_tx = False

# Generate towers with x and y coordinates.
# for tower i: x, y = towers[i][0], towers[i][1]
towersss = (np.random.rand(num_towers, 3) - 0.5) * rx_square_side
array_2dd = np.array(towersss)
zahl1 = [1, 1, 1, 1]
array_1dd = np.array(zahl1)
towerss = array_2dd * array_1dd[:, np.newaxis]
zahl = [1, 1, 0]
array_2d = np.array(towerss)
array_1d = np.array(zahl)
towers = array_2d * array_1d[np.newaxis, :]
print('towers:\n', towers)

# location of transmitting device with tx[0] being x and tx[1] being y.
tx = (np.random.rand(3) - [0.5, 0.5, -1]) * tx_square_side

print('tx:', tx)

# Distances from each tower to the transmitting device,
# simply triangle hypotenuse.
# distances[i] is distance from tower i to transmitter.
distances = np.array([np.sqrt((x[0] - tx[0]) ** 2 + (x[1] - tx[1]) ** 2 + (x[2] - tx[2]) ** 2)
                      for x in towers])
print('distances:', distances)

# Time at which each tower receives the transmission.
rec_times = distances / v
# Add noise to receive times
rec_times += np.random.normal(loc=0, scale=rec_time_noise_stdd,
                              size=num_towers)
print('rec_times:', rec_times)

"""
## Create a visualisation for TDOA
"""

# Plot towers and transmission location.
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
max_width = max(tx_square_side, rx_square_side) / 3
zlim = ax.set_zlim((max_width * -1, max_width))
ylim = ax.set_ylim((max_width * -1, max_width))
xlim = ax.set_xlim((max_width * -1, max_width))
ax.plot((0, 0), (0, 0), (-max_width + 1, max_width - 1), 'b', label='z-axis')
ax.plot((-max_width + 1, max_width - 1), (0, 0), (0, 0), 'r', label='x-axis')
ax.plot((0, 0), (-max_width + 1, max_width - 1), (0, 0), 'k', label='y-axis')

x0, y0, z0 = [], [], []
for i in range(0, len(towers)):
    x0.__iadd__([towers[i][0]])
    y0.__iadd__([towers[i][1]])
    z0.__iadd__([towers[i][2]])

r = []
for i in range(0, len(distances)):
    r.__iadd__([distances[i]])

if plot_trilateration_tx:
    # Iterate over every unique combination of towers and plot nifty stuff.
    for i in range(num_towers):
        if plot_lines_to_tx:
            # line between transmitter and tx
            ax.plot3D((towers[i][0], tx[0]),
                      (towers[i][1], tx[1]),
                      (towers[i][2], tx[2]))

        for j in range(i + 1, num_towers):
            if plot_lines_between_towers:
                # Line between towers
                ax.plot3D((towers[i][0], towers[j][0]),
                          (towers[i][1], towers[j][1]),
                          (towers[i][2], towers[j][2]))


    def plot_trilateration_tx():
        global v_vec, v_ann, t0_rec, t1_rec, t2_rec, t3_rec, TDOA1, TDOA2, TDOA3, TDOA4, TDOA5, TDOA6
        # Kugeln
        Theta, Phi = np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 40)
        theta1, phi = np.meshgrid(Theta, Phi)
        radius = 1e3

        def plot_tx(radius):
            X1 = tx[0] + radius * np.sin(phi) * np.cos(theta1)
            Y1 = tx[1] + radius * np.sin(phi) * np.sin(theta1)
            Z1 = tx[2] + radius * np.cos(phi)
            plot_trilateration_tx_plot = ax.plot_surface(
                X1, Y1, Z1, rstride=1, cstride=1, cmap=cm.coolwarm,
                linewidth=0, antialiased=False, alpha=0.3)

        def plot_towers():
            for k in range(towers.shape[0]):
                x = towers[k][0]
                y = towers[k][1]
                z = towers[k][2]
                ax.text3D(x, y, z, 'Tower ' + str(k))
                ax.text3D(tx[0], tx[1], tx[2], 'Tx')
                pp1 = ax.scatter3D(x, y, z, color="b", s=10)
                pp2 = ax.scatter3D(tx[0], tx[1], tx[2], color="g", s=10)

        # Annotations, to be updated during animation
        cur_time = ax.text(0, 30e3, 26e3, 't = 0')
        t0 = ax.text(0, 30e3, 23e3, 'Tower 0 received at t = ')
        t1 = ax.text(0, 30e3, 20e3, 'Tower 1 received at t = ')
        t2 = ax.text(0, 30e3, 17e3, 'Tower 2 received at t = ')
        t3 = ax.text(0, 30e3, 14e3, 'Tower 3 received at t = ')
        TDOA1_ann = ax.text(0, 30e3, 11e3, 'TDOA1 ')
        TDOA2_ann = ax.text(0, 30e3, 8e3, 'TDOA2 ')
        TDOA3_ann = ax.text(0, 30e3, 5e3, 'TDOA3 ')
        TDOA4_ann = ax.text(0, 30e3, 3e3, 'TDOA4 ')
        TDOA5_ann = ax.text(0, 30e3, 2e3, 'TDOA5 ')
        TDOA6_ann = ax.text(0, 30e3, 1e3, 'TDOA6 ')
        v_vec = ax.quiver(tx[0], tx[1] + 1e3, tx[2], 0, 1, 0,
                         length=2500, normalize=True, fc='k', ec='k')

        v_ann = ax.text3D(tx[0], tx[1] + 1e3, tx[2], 'v = {:.0E} m/s'.format(v))

        n_frames = 10
        max_seconds = 100e-6
        t0_rec = 0
        t1_rec = 0
        t2_rec = 0
        t3_rec = 0
        TDOA1 = 0
        TDOA2 = 0
        TDOA3 = 0
        TDOA4 = 0
        TDOA5 = 0
        TDOA6 = 0

        def animate(i):
            global t0_rec, t1_rec, t2_rec, t3_rec, TDOA1, TDOA2, TDOA3, TDOA4, TDOA5, TDOA6, v_vec, v_ann,TDOA_dist1,\
                TDOA_dist2,TDOA_dist3,TDOA_dist4,TDOA_dist5,TDOA_dist6

            t = i / n_frames * max_seconds
            Radius = v * t

            v_vec.remove()
            ax.collections.clear()

            v_vec = ax.quiver(tx[0], tx[1] + Radius, tx[2], 0, 1, 0,
                                  length=2500, normalize=True, fc='k', ec='k')

            v_ann.set_position((tx[0] - 0.5e3, tx[1] + 1.5e3 + Radius))
            plot_towers()
            plot_tx(radius=Radius)

            cur_time.set_text('t = {:.2E} s'.format(t))
            if t >= rec_times[0] and t0_rec == 0:
                t0_rec = t
                t0.set_text(t0.get_text() + '{:.2E} s'.format(t0_rec))
            if t >= rec_times[1] and t1_rec == 0:
                t1_rec = t
                t1.set_text(t1.get_text() + '{:.2E} s'.format(t1_rec))
            if t >= rec_times[2] and t2_rec == 0:
                t2_rec = t
                t2.set_text(t2.get_text() + '{:.2E} s'.format(t2_rec))
            if t >= rec_times[3] and t3_rec == 0:
                t3_rec = t
                t3.set_text(t3.get_text() + '{:.2E} s'.format(t3_rec))
            if t0_rec != 0 and t1_rec != 0 and TDOA1 == 0:
                TDOA1 = t1_rec - t0_rec
                TDOA_dist1 = v * TDOA1
                TDOA1_ann.set_text(TDOA1_ann.get_text() + 't1-t0 =' + '{:.2E} s'.format(TDOA1))
            if t0_rec != 0 and t2_rec != 0 and TDOA2 == 0:
                TDOA2 = abs(t2_rec - t0_rec)
                TDOA_dist2 = v * TDOA2
                TDOA2_ann.set_text(TDOA2_ann.get_text() + 't2-t0 =' + '{:.2E} s'.format(TDOA2))
            if t0_rec != 0 and t3_rec != 0 and TDOA3 == 0:
                TDOA3 = abs(t3_rec - t0_rec)
                TDOA_dist3 = v * TDOA3
                TDOA3_ann.set_text(TDOA3_ann.get_text() + 't3-t0 =' + '{:.2E} s'.format(TDOA3))
            if t1_rec != 0 and t3_rec != 0 and TDOA4 == 0:
                TDOA4 = abs(t3_rec - t1_rec)
                TDOA_dist4 = v * TDOA4
                TDOA4_ann.set_text(TDOA4_ann.get_text() + 't3-t1 =' + '{:.2E} s'.format(TDOA4))
            if t2_rec != 0 and t3_rec != 0 and TDOA5 == 0:
                TDOA5 = abs(t3_rec - t2_rec)
                TDOA_dist5 = v * TDOA5
                TDOA5_ann.set_text(TDOA5_ann.get_text() + 't3-t2 =' + '{:.2E} s'.format(TDOA5))
            if t1_rec != 0 and t2_rec != 0 and TDOA6 == 0:
                TDOA6 = abs(t2_rec - t1_rec)
                TDOA_dist6 = v * TDOA6
                TDOA6_ann.set_text(TDOA6_ann.get_text() + 't2-t2 =' + '{:.2E} s'.format(TDOA6))

        anim = FuncAnimation(fig, animate,frames=n_frames, interval=16.6, blit=False)
        # anim.save('C:/Users/Mem/Desktop/Studium/Vertiefungsmodul/Animationen/TDOA.gif', writer='imagemagick', fps=60)
        # plt.close()
        # Image(url='C:/Users/Mem/Desktop/Studium/Vertiefungsmodul/Animationen/TDOA.gif')
        plt.show()
        return anim


    plot_trilateration_tx()

"""
## Create a visualisation for locus
"""
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')

if plot_trilateration_spheresIntersection_circles:
    for i in range(num_towers):
        if plot_lines_to_tx:
            # line between transmitter and tx
            ax.plot3D((towers[i][0], tx[0]),
                      (towers[i][1], tx[1]),
                      (towers[i][2], tx[2]))

        for j in range(i + 1, num_towers):
            if plot_lines_between_towers:
                # Line between towers
                ax.plot3D((towers[i][0], towers[j][0]),
                          (towers[i][1], towers[j][1]),
                          (towers[i][2], towers[j][2]))


    def plot_trilateration_spheresIntersection():
        # Kugeln
        Theta, Phi = np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 40)
        theta1, phi = np.meshgrid(Theta, Phi)


        class Kugeln:
            def __init__(self, radius, x, y, z):
                self.radius = radius
                self.x = x
                self.y = y
                self.z = z

            def coordinaten(self):
                X = self.x + self.radius * np.sin(phi) * np.cos(theta1)
                Y = self.y + self.radius * np.sin(phi) * np.sin(theta1)
                Z = self.z + self.radius * np.cos(phi)
                plot = ax.plot_surface(
                    X, Y, Z, rstride=2, cstride=2, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False, alpha=0.2)
                return plot

        def kugel1(r1):
            X2 = x0[0] + r1 * np.sin(phi) * np.cos(theta1)
            Y2 = y0[0] + r1 * np.sin(phi) * np.sin(theta1)
            Z2 = z0[0] + r1 * np.cos(phi)
            ax.plot_surface(
                X2, Y2, Z2, rstride=2, cstride=2, cmap=cm.coolwarm,
                linewidth=0, antialiased=False, alpha=0.2)

        def kugel2(r2):
            X2 = x0[1] + r2 * np.sin(phi) * np.cos(theta1)
            Y2 = y0[1] + r2 * np.sin(phi) * np.sin(theta1)
            Z2 = z0[1] + r2 * np.cos(phi)
            ax.plot_surface(
                X2, Y2, Z2, rstride=2, cstride=2, cmap=cm.coolwarm,
                linewidth=0, antialiased=False, alpha=0.2)

        def kugel3(r3):
            X2 = x0[2] + r3 * np.sin(phi) * np.cos(theta1)
            Y2 = y0[2] + r3 * np.sin(phi) * np.sin(theta1)
            Z2 = z0[2] + r3 * np.cos(phi)
            ax.plot_surface(
                X2, Y2, Z2, rstride=2, cstride=2, cmap=cm.coolwarm,
                linewidth=0, antialiased=False, alpha=0.2)

        def kugel4(r4):
            X2 = x0[3] + r4 * np.sin(phi) * np.cos(theta1)
            Y2 = y0[3] + r4 * np.sin(phi) * np.sin(theta1)
            Z2 = z0[3] + r4 * np.cos(phi)
            ax.plot_surface(
                X2, Y2, Z2, rstride=1, cstride=2, cmap=cm.coolwarm,
                linewidth=0, antialiased=False, alpha=0.2)

        d = x0[1] - x0[0]
        g = y0[1] - y0[0]
        h = z0[1] - z0[0]
        e = x0[2] - x0[0]
        f = y0[2] - y0[0]
        n = z0[2] - z0[0]
        a = x0[3] - x0[0]
        b = y0[3] - y0[0]
        c = z0[3] - z0[0]
        x, y, z = sy.symbols("x y z")
        equations = [
            sy.Eq((x - x0[0]) ** 2 + (y - y0[0]) ** 2 + (z - z0[0]) ** 2, r[0] ** 2),
            sy.Eq((x - d - x0[0]) ** 2 + (y - g - y0[0]) ** 2 + (z - h - z0[0]) ** 2, r[1] ** 2),
            sy.Eq((x - e - x0[0]) ** 2 + (y - f - y0[0]) ** 2 + (z - n - z0[0]) ** 2, r[2] ** 2),
            sy.Eq((x - a - x0[0]) ** 2 + (y - b - y0[0]) ** 2 + (z - c - z0[0]) ** 2, r[3] ** 2),
        ]
        if np.sqrt(d ** 2 + g ** 2 + h ** 2) <= (r[0] + r[1]) and \
                np.sqrt(e ** 2 + f ** 2 + n ** 2) <= (r[0] + r[2]) and \
                np.sqrt(a ** 2 + b ** 2 + c ** 2) <= (r[0] + r[3]) and \
                np.sqrt((abs(d) - abs(e)) ** 2 + (abs(g) - abs(f)) ** 2 + (abs(h) - abs(n)) ** 2) <= (
                r[1] + r[2]) and \
                np.sqrt((abs(d) - abs(a)) ** 2 + (abs(g) - abs(b)) ** 2 + (abs(h) - abs(c)) ** 2) <= (
                r[1] + r[3]) and \
                np.sqrt((abs(e) - abs(a)) ** 2 + (abs(f) - abs(b)) ** 2 + (abs(n) - abs(c)) ** 2) <= (r[2] + r[3]):
            solved = sy.solve(equations)
            if np.shape(solved) > (0,):
                lsg1 = solved[0]
                data1 = list(lsg1.values())
                ax.text(1e3, 30e3, 33e3, f'The first location of the point is: {data1}')
                if len(solved) == 2:
                    lsg2 = solved[1]
                    data2 = list(lsg2.values())
                    ax.text(1e3, 30e3, 31e3, f'The second location of the point is: {data2}')
                else:
                    data2 = data1
                # Treffpunkt
                eqPoint1 = ax.scatter3D(data1[0], data1[1], data1[2])
                eqPoint2 = ax.scatter3D(data2[0], data2[1], data2[2])
            else:
                posi = Trilateration_3D(towers, distances)
                ax.text(1e3, 30e3, 29e3,
                        f'there is no sy.solve for the intesection, but Trilateration_3D has one:{posi}')
                # eqPoint1 = ax.scatter3D(posi[0], posi[1], posi[2])
                # eqPoint2 = eqPoint1
        else:
            ax.text(1e3, 30e3, 27e3,
                    f'there is no Intersection')
            eqPoint1 = [0, 0, 0]
            eqPoint2 = [0, 0, 0]

        theta, te = np.linspace(0, 2 * np.pi, 80), np.linspace(0, 2 * np.pi, 80)

        def circle12(radius):
            if np.sqrt(d ** 2 + g ** 2 + h ** 2) <= (r[0] + r[1]):
                X12 = (d ** 2 + g ** 2 + h ** 2 + r[0] ** 2 - r[1] ** 2) / (
                        2 * np.sqrt(d ** 2 + g ** 2 + h ** 2) * r[0])
                coord = sphereCircle(radius, 0, math.atan2(g, d), np.arccos(X12), theta)
                array_2d = np.array(coord)
                Zahl = [x0[0], y0[0], z0[0]]
                array_1d = np.array(Zahl)
                coordinats_12 = array_2d + array_1d[:, np.newaxis]
                plot_circle_1 = ax.plot3D(
                    coordinats_12[0], coordinats_12[1], coordinats_12[2], color="g")
            else:
                plot_circle_1 = []
        def circle13(radius):
            if np.sqrt(e ** 2 + f ** 2 + n ** 2) <= (r[0] + r[2]):
                X13 = (r[0] ** 2 - r[2] ** 2 + e ** 2 + f ** 2 + n ** 2) / (
                        2 * r[0] * np.sqrt(e ** 2 + f ** 2 + n ** 2))
                coord = sphereCircle(radius, 0, math.atan2(f, e), np.arccos(X13), te)
                array_2d = np.array(coord)
                Zahl = [x0[0], y0[0], z0[0]]
                array_1d = np.array(Zahl)
                coordinats_13 = array_2d + array_1d[:, np.newaxis]
                plot_circle_2 = ax.plot(
                    coordinats_13[0], coordinats_13[1], coordinats_13[2])
            else:
                plot_circle_2 = []

        def circle14(radius):
            if np.sqrt(a ** 2 + b ** 2 + c ** 2) <= (r[0] + r[3]):
                X14 = (r[0] ** 2 - r[3] ** 2 + a ** 2 + b ** 2 + c ** 2) / (
                        2 * r[0] * np.sqrt(a ** 2 + b ** 2 + c ** 2))
                coord = sphereCircle(radius, math.atan2(c, np.sqrt(a ** 2 + b ** 2)), math.atan2(b, a),
                                     np.arccos(X14),
                                     te)
                array_2d = np.array(coord)
                Zahl = [x0[0], y0[0], z0[0]]
                array_1d = np.array(Zahl)
                coordinats_14 = array_2d + array_1d[:, np.newaxis]
                plot_circle_4 = ax.plot(
                    coordinats_14[0], coordinats_14[1], coordinats_14[2])
            else:
                plot_circle_4 = []

        def asec(x):
            if x < -1 or x > 1:
                return math.acos(1 / x)
            else:
                return 0

        def circle23(radius):
            if np.sqrt((abs(d) - abs(e)) ** 2 + (abs(g) - abs(f)) ** 2 + (abs(h) - abs(n)) ** 2) <= (r[1] + r[2]):
                X23 = 2 * np.sqrt((d - e) ** 2 + (f - g) ** 2 + (n - h) ** 2) * r[1] / (
                        (d - e) ** 2 + (f - g) ** 2 + (n - h) ** 2 + r[1] ** 2 - r[2] ** 2)
                coord = sphereCircle(radius, np.pi - math.atan2(n - h, np.sqrt((d - e) ** 2 + (g - f) ** 2)),
                                     (math.atan2(g - f, d - e)),
                                     asec(X23), te)
                array_2d = np.array(coord)
                Zahl = [d + x0[0], g + y0[0], h + z0[0]]
                array_1d = np.array(Zahl)
                coordinats_23 = array_2d + array_1d[:, np.newaxis]
                plot_circle_3 = ax.plot(
                    coordinats_23[0], coordinats_23[1], coordinats_23[2])
            else:
                plot_circle_3 = []

        def circle24(radius):
            if np.sqrt((abs(d) - abs(a)) ** 2 + (abs(g) - abs(b)) ** 2 + (abs(h) - abs(c)) ** 2) <= (r[1] + r[3]):
                X24 = 2 * np.sqrt((d - a) ** 2 + (b - g) ** 2 + (c - h) ** 2) * r[1] / (
                        (d - a) ** 2 + (g - b) ** 2 + (h - c) ** 2 + r[1] ** 2 - r[3] ** 2)
                coord = sphereCircle(radius, np.pi - math.atan2(c - h, np.sqrt((d - a) ** 2 + (g - b) ** 2)),
                                     (math.atan2(g - b, d - a)),
                                     asec(X24), te)
                array_2d = np.array(coord)
                Zahl = [d + x0[0], g + y0[0], h + z0[0]]
                array_1d = np.array(Zahl)
                coordinats_24 = array_2d + array_1d[:, np.newaxis]
                plot_circle_5 = ax.plot(
                    coordinats_24[0], coordinats_24[1], coordinats_24[2])
            else:
                plot_circle_5 = []

        def circle34(radius):
            if np.sqrt((abs(e) - abs(a)) ** 2 + (abs(f) - abs(b)) ** 2 + (abs(n) - abs(c)) ** 2) <= (r[2] + r[3]):
                X34 = 2 * np.sqrt((e - a) ** 2 + (f - b) ** 2 + (c - n) ** 2) * r[2] / (
                        (e - a) ** 2 + (f - b) ** 2 + (c - n) ** 2 + r[2] ** 2 - r[3] ** 2)
                coord = sphereCircle(radius, np.pi - math.atan2(c - n, np.sqrt((e - a) ** 2 + (f - b) ** 2)),
                                     (math.atan2(f - b, e - a)),
                                     asec(X34), te)
                array_2d = np.array(coord)
                Zahl = [e + x0[0], f + y0[0], n + z0[0]]
                array_1d = np.array(Zahl)
                coordinats_34 = array_2d + array_1d[:, np.newaxis]
                plot_circle_6 = ax.plot(
                    coordinats_34[0], coordinats_34[1], coordinats_34[2])
            else:
                plot_circle_6 = []

        def plot_towers():
            for k in range(towers.shape[0]):
                x = towers[k][0]
                y = towers[k][1]
                z = towers[k][2]
                ax.text3D(x, y, z, 'Tower ' + str(k))
                ax.text3D(tx[0], tx[1], tx[2], 'Tx')
                ax.scatter3D(x, y, z, color="b", s=10)
                ax.scatter3D(tx[0], tx[1], tx[2], color="g", s=10)



        n_frames = 100
        max_d = 1e5

        def animate(i):
            global TDOA_dist1, TDOA_dist2, TDOA_dist3, TDOA_dist4, TDOA_dist5, TDOA_dist6

            d = np.sqrt(i) / n_frames * max_d
            d0 = np.clip(d, i / n_frames * max_d, distances[0])
            d1 = np.clip(d, i / n_frames * max_d, distances[1])
            d2 = np.clip(d, i / n_frames * max_d, distances[2])
            d3 = np.clip(d, i / n_frames * max_d, distances[3])

            r1 = np.clip(d, i / n_frames * max_d, r[0])
            r2 = np.clip(d, i / n_frames * max_d, r[1])
            r3 = np.clip(d, i / n_frames * max_d, r[2])

            ax.clear()
            ax.collections.clear()

            max_width = max(tx_square_side, rx_square_side) / 3
            ax.set_zlim((max_width * -1, max_width))
            ax.set_ylim((max_width * -1, max_width))
            ax.set_xlim((max_width * -1, max_width))
            ax.plot((0, 0), (0, 0), (-max_width + 1, max_width - 1), 'b', label='z-axis')
            ax.plot((-max_width + 1, max_width - 1), (0, 0), (0, 0), 'r', label='x-axis')
            ax.plot((0, 0), (-max_width + 1, max_width - 1), (0, 0), 'k', label='y-axis')


            plot_towers()

            kugel1 = Kugeln(radius=d0, x=x0[0], y=y0[0], z=z0[0])
            kugel1.coordinaten()
            kugel2 = Kugeln(radius=d1, x=x0[1], y=y0[1], z=z0[1])
            kugel2.coordinaten()

            # kugel1(r1=d0)
            # kugel2(r2=d1 + TDOA_dist1)
            # kugel3(r3=d2 + TDOA_dist2)
            # kugel4(r4=d3 + TDOA_dist3

            circle12(radius=r1)
            # circle13(radius=r1 + TDOA_dist2)
            # circle14(radius=r1 + TDOA_dist3)
            # circle23(radius=r2 + TDOA_dist6)
            # circle24(radius=r2 + TDOA_dist4)
            # circle34(radius=r3 + TDOA_dist5)


            # Annotations, to be updated during animation
            cur_d_ann = ax.text(1e3, 30e3, 25e3, 'd = 0')
            r0_ann = ax.text(1e3, 30e3, 23e3, 'Tower 0 radius r0 = ')
            r1_ann = ax.text(1e3, 30e3, 21e3, 'Tower 1 radius r1 = ')
            cur_d_ann.set_text('d = {:.2E} m'.format(d))
            r0_ann.set_text('Tower 0 radius r0 = {:.2E} m'.format(d0))
            r1_ann.set_text('Tower 1 radius r1 = {:.2E} m'.format(d1 + TDOA_dist1))
            r1_ann.set_text('Tower 1 radius r1 = {:.2E} m'.format(d2 + TDOA_dist2))
            r1_ann.set_text('Tower 1 radius r1 = {:.2E} m'.format(d3 + TDOA_dist3))

        anim = FuncAnimation(fig, animate, frames=60, interval=16, blit=False)
        # anim.save('C:/Users/Mem/Desktop/Studium/Vertiefungsmodul/Animationen/TDOA.gif', writer='imagemagick', fps=60)
        #anim.save('/home/soeren/Animations/TDOA.gif', writer='imagemagick', fps=60)
        #plt.close()
        # Image(url='C:/Users/Mem/Desktop/Studium/Vertiefungsmodul/Animationen/TDOA.gif')
        plt.show()
        return anim


    plot_trilateration_spheresIntersection()


def sfff():
    pass


def plot_trilateration_Intersection():
    d = x0[1] - x0[0]
    g = y0[1] - y0[0]
    h = z0[1] - z0[0]
    e = x0[2] - x0[0]
    f = y0[2] - y0[0]
    n = z0[2] - z0[0]
    a = x0[3] - x0[0]
    b = y0[3] - y0[0]
    c = z0[3] - z0[0]
    x, y, z = sy.symbols("x y z")
    equations = [
        sy.Eq((x - x0[0]) ** 2 + (y - y0[0]) ** 2 + (z - z0[0]) ** 2, r[0] ** 2),
        sy.Eq((x - d - x0[0]) ** 2 + (y - g - y0[0]) ** 2 + (z - h - z0[0]) ** 2, r[1] ** 2),
        sy.Eq((x - e - x0[0]) ** 2 + (y - f - y0[0]) ** 2 + (z - n - z0[0]) ** 2, r[2] ** 2),
        sy.Eq((x - a - x0[0]) ** 2 + (y - b - y0[0]) ** 2 + (z - c - z0[0]) ** 2, r[3] ** 2),
    ]
    if np.sqrt(d ** 2 + g ** 2 + h ** 2) <= (r[0] + r[1]) and \
            np.sqrt(e ** 2 + f ** 2 + n ** 2) <= (r[0] + r[2]) and \
            np.sqrt(a ** 2 + b ** 2 + c ** 2) <= (r[0] + r[3]) and \
            np.sqrt((abs(d) - abs(e)) ** 2 + (abs(g) - abs(f)) ** 2 + (abs(h) - abs(n)) ** 2) <= (
            r[1] + r[2]) and \
            np.sqrt((abs(d) - abs(a)) ** 2 + (abs(g) - abs(b)) ** 2 + (abs(h) - abs(c)) ** 2) <= (
            r[1] + r[3]) and \
            np.sqrt((abs(e) - abs(a)) ** 2 + (abs(f) - abs(b)) ** 2 + (abs(n) - abs(c)) ** 2) <= (r[2] + r[3]):
        solved = sy.solve(equations)
        if np.shape(solved) > (0,):
            lsg1 = solved[0]
            data1 = list(lsg1.values())
            location = print(f'The first location of the point is: {data1}')
            if len(solved) == 2:
                lsg2 = solved[1]
                data2 = list(lsg2.values())
                location = print(f'The second location of the point is: {data2}')
            else:
                data2 = data1
            # Treffpunkt
            eqPoint1 = ax.scatter3D(data1[0], data1[1], data1[2])
            eqPoint2 = ax.scatter3D(data2[0], data2[1], data2[2])
        else:
            posi = Trilateration_3D(towers, distances)
            location = print(f'there is no sy.solve for the intesection, but Trilateration_3D has one: {posi}')
            # eqPoint1 = ax.scatter3D(posi[0], posi[1], posi[2])
            # eqPoint2 = eqPoint1
    else:
        location = print(f'there is no intesection')
        eqPoint1 = [0, 0, 0]
        eqPoint2 = [0, 0, 0]
    theta, te = np.linspace(0, 2 * np.pi, 80), np.linspace(0, 2 * np.pi, 80)
    if np.sqrt(d ** 2 + g ** 2 + h ** 2) <= (r[0] + r[1]):
        X12 = (d ** 2 + g ** 2 + h ** 2 + r[0] ** 2 - r[1] ** 2) / (
                2 * np.sqrt(d ** 2 + g ** 2 + h ** 2) * r[0])
        coord = sphereCircle(r[0], 0, math.atan2(g, d), np.arccos(X12), theta)
        array_2d = np.array(coord)
        Zahl = [x0[0], y0[0], z0[0]]
        array_1d = np.array(Zahl)
        coordinats_12 = array_2d + array_1d[:, np.newaxis]
        plot_circle_1 = ax.plot(
            coordinats_12[0], coordinats_12[1], coordinats_12[2])
    else:
        plot_circle_1 = []
    if np.sqrt(e ** 2 + f ** 2 + n ** 2) <= (r[0] + r[2]):
        X13 = (r[0] ** 2 - r[2] ** 2 + e ** 2 + f ** 2 + n ** 2) / (
                2 * r[0] * np.sqrt(e ** 2 + f ** 2 + n ** 2))
        coord = sphereCircle(r[0], 0, math.atan2(f, e), np.arccos(X13), te)
        array_2d = np.array(coord)
        Zahl = [x0[0], y0[0], z0[0]]
        array_1d = np.array(Zahl)
        coordinats_13 = array_2d + array_1d[:, np.newaxis]
        plot_circle_2 = ax.plot(
            coordinats_13[0], coordinats_13[1], coordinats_13[2])
    else:
        plot_circle_2 = []

    def asec(x):
        if x < -1 or x > 1:
            return math.acos(1 / x)
        else:
            return 0

    if np.sqrt((abs(d) - abs(e)) ** 2 + (abs(g) - abs(f)) ** 2 + (abs(h) - abs(n)) ** 2) <= (r[1] + r[2]):
        X23 = 2 * np.sqrt((d - e) ** 2 + (f - g) ** 2 + (n - h) ** 2) * r[1] / (
                (d - e) ** 2 + (f - g) ** 2 + (n - h) ** 2 + r[1] ** 2 - r[2] ** 2)
        coord = sphereCircle(r[1], np.pi - math.atan2(n - h, np.sqrt((d - e) ** 2 + (g - f) ** 2)),
                             (math.atan2(g - f, d - e)),
                             asec(X23), te)
        array_2d = np.array(coord)
        Zahl = [d + x0[0], g + y0[0], h + z0[0]]
        array_1d = np.array(Zahl)
        coordinats_23 = array_2d + array_1d[:, np.newaxis]
        plot_circle_3 = ax.plot(
            coordinats_23[0], coordinats_23[1], coordinats_23[2])
    else:
        plot_circle_3 = []
    if np.sqrt(a ** 2 + b ** 2 + c ** 2) <= (r[0] + r[3]):
        X14 = (r[0] ** 2 - r[3] ** 2 + a ** 2 + b ** 2 + c ** 2) / (
                2 * r[0] * np.sqrt(a ** 2 + b ** 2 + c ** 2))
        coord = sphereCircle(r[0], math.atan2(c, np.sqrt(a ** 2 + b ** 2)), math.atan2(b, a), np.arccos(X14),
                             te)
        array_2d = np.array(coord)
        Zahl = [x0[0], y0[0], z0[0]]
        array_1d = np.array(Zahl)
        coordinats_14 = array_2d + array_1d[:, np.newaxis]
        plot_circle_4 = ax.plot(
            coordinats_14[0], coordinats_14[1], coordinats_14[2])
    else:
        plot_circle_4 = []
    if np.sqrt((abs(d) - abs(a)) ** 2 + (abs(g) - abs(b)) ** 2 + (abs(h) - abs(c)) ** 2) <= (r[1] + r[3]):
        X24 = 2 * np.sqrt((d - a) ** 2 + (b - g) ** 2 + (c - h) ** 2) * r[1] / (
                (d - a) ** 2 + (g - b) ** 2 + (h - c) ** 2 + r[1] ** 2 - r[3] ** 2)
        coord = sphereCircle(r[1], np.pi - math.atan2(c - h, np.sqrt((d - a) ** 2 + (g - b) ** 2)),
                             (math.atan2(g - b, d - a)),
                             asec(X24), te)
        array_2d = np.array(coord)
        Zahl = [d + x0[0], g + y0[0], h + z0[0]]
        array_1d = np.array(Zahl)
        coordinats_24 = array_2d + array_1d[:, np.newaxis]
        plot_circle_5 = ax.plot(
            coordinats_24[0], coordinats_24[1], coordinats_24[2])
    else:
        plot_circle_5 = []
    if np.sqrt((abs(e) - abs(a)) ** 2 + (abs(f) - abs(b)) ** 2 + (abs(n) - abs(c)) ** 2) <= (r[2] + r[3]):
        X34 = 2 * np.sqrt((e - a) ** 2 + (f - b) ** 2 + (c - n) ** 2) * r[2] / (
                (e - a) ** 2 + (f - b) ** 2 + (c - n) ** 2 + r[2] ** 2 - r[3] ** 2)
        coord = sphereCircle(r[2], np.pi - math.atan2(c - n, np.sqrt((e - a) ** 2 + (f - b) ** 2)),
                             (math.atan2(f - b, e - a)),
                             asec(X34), te)
        array_2d = np.array(coord)
        Zahl = [e + x0[0], f + y0[0], n + z0[0]]
        array_1d = np.array(Zahl)
        coordinats_34 = array_2d + array_1d[:, np.newaxis]
        plot_circle_6 = ax.plot(
            coordinats_34[0], coordinats_34[1], coordinats_34[2])
    else:
        plot_circle_6 = []
