from mpl_toolkits.mplot3d import *
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import sympy as sy
import math
import sys
import time
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
# reception times at the towers. Mean is zero.
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
plot_lines_to_tx = True

# Generate towers with x and y coordinates.
# for tower i: x, y = towers[i][0], towers[i][1]
towersss = (np.random.rand(num_towers, 3) - 0.5) * rx_square_side
array_2dd = np.array(towersss)
zahl1 = np.repeat(1, num_towers)
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


x0, y0, z0 = [], [], []
for i in range(towers.shape[0]):
    x0.__iadd__([towers[i][0]])
    y0.__iadd__([towers[i][1]])
    z0.__iadd__([towers[i][2]])

r = []
for i in range(towers.shape[0]):
    r.__iadd__([distances[i]])


if plot_trilateration_spheresIntersection_circles:

    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(projection='3d')
    fig.subplots_adjust(left=0.2, right=0.8)
    max_width = max(tx_square_side, rx_square_side) / 2
    ax.set_zlim((max_width * -2, max_width*2))
    ax.set_ylim((max_width * -2, max_width*2))
    ax.set_xlim((max_width * -2, max_width*2))
    ax.plot((0, 0), (0, 0), (-max_width + 1, max_width - 1), 'b', label='z-axis')
    ax.plot((-max_width + 1, max_width - 1), (0, 0), (0, 0), 'r', label='x-axis')
    ax.plot((0, 0), (-max_width + 1, max_width - 1), (0, 0), 'k', label='y-axis')
    plt.axis('off')

    def plot_lines():
        for i in range(num_towers):
            if plot_lines_to_tx:
                # line between transmitter and tx
                pl1 = ax.plot3D((towers[i][0], tx[0]),
                          (towers[i][1], tx[1]),
                          (towers[i][2], tx[2]))

            for j in range(i + 1, num_towers):
                if plot_lines_between_towers:
                    # Line between towers
                    pl2 = ax.plot3D((towers[i][0], towers[j][0]),
                              (towers[i][1], towers[j][1]),
                              (towers[i][2], towers[j][2]))



    # Kugeln
    Theta, Phi = np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 40)
    theta1, phi = np.meshgrid(Theta, Phi)
    def plot_tx(radius):
        X1 = tx[0] + radius * np.sin(phi) * np.cos(theta1)
        Y1 = tx[1] + radius * np.sin(phi) * np.sin(theta1)
        Z1 = tx[2] + radius * np.cos(phi)
        plot_trilateration_tx_plot = ax.plot_surface(
            X1, Y1, Z1, rstride=1, cstride=1, cmap=cm.coolwarm,
            linewidth=0, antialiased=False, alpha=0.3)
        return plot_trilateration_tx_plot
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
            print(f'The first location of the point is: {data1}')
            if len(solved) == 2:
                lsg2 = solved[1]
                data2 = list(lsg2.values())
                print(f'The second location of the point is: {data2}')
            else:
                data2 = data1
            # Treffpunkt
            eqPoint1 = ax.scatter3D(data1[0], data1[1], data1[2])
            eqPoint2 = ax.scatter3D(data2[0], data2[1], data2[2])
        else:
            posi = Trilateration_3D(towers, distances)
            print(f'there is no sy.solve for the intesection, but Trilateration_3D has one:{posi}')
            # eqPoint1 = ax.scatter3D(posi[0], posi[1], posi[2])
            # eqPoint2 = eqPoint1
    else:
        print(f'there is no Intersection')
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
            plot_circle_1 = ax.plot(
                coordinats_12[0], coordinats_12[1], coordinats_12[2],'g-')
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
                coordinats_13[0], coordinats_13[1], coordinats_13[2],'g-')
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
                coordinats_14[0], coordinats_14[1], coordinats_14[2],'g-')
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
                coordinats_23[0], coordinats_23[1], coordinats_23[2],'g-')
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
                coordinats_24[0], coordinats_24[1], coordinats_24[2],'g-')
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
                coordinats_34[0], coordinats_34[1], coordinats_34[2],'g-')
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

    # Annotations, to be updated during animation
    l = 40e3
    k = 3e3
    cur_time = ax.text(0, 30e3, l + k, 't = 0')
    tower_text = []
    for i in range(num_towers):
        text = ax.text(0, 30e3, -i * k + l, 'Tower {} received at t = '.format(i))
        tower_text.append(text)

    v_vec = ax.quiver(tx[0], tx[1] + 1e3, tx[2], 0, 1, 0,
                      length=2500, normalize=True, fc='k', ec='k')
    v_ann = ax.text3D(tx[0], tx[1] + 1e3, tx[2], 'v = {:.0E} m/s'.format(v))
    t_rec = 0
    TDOA1 = abs(rec_times[1] - rec_times[0])
    TDOA2 = abs(rec_times[2] - rec_times[0])
    TDOA3 = abs(rec_times[3] - rec_times[0])
    TDOA4 = abs(rec_times[3] - rec_times[1])
    TDOA5 = abs(rec_times[3] -

 rec_times[2])
    TDOA6 = abs(rec_times[2] - rec_times[1])
    TDOA1_dist = v * TDOA1
    TDOA2_dist = v * TDOA2
    TDOA3_dist = v * TDOA3
    TDOA4_dist = v * TDOA4
    TDOA5_dist = v * TDOA5
    TDOA6_dist = v * TDOA6

    n_frames = 10
    max_seconds = 1e-4
    max_d = 1e5

    def animate1(i):
        global t_rec, tower_text, TDOA1, TDOA2, TDOA3, TDOA4, TDOA5, TDOA6, v_vec, v_ann, TDOA1_dist, \
            TDOA2_dist, TDOA3_dist, TDOA4_dist, TDOA5_dist, TDOA6_dist

        t = i / n_frames * max_seconds
        Radius = v * t

        ax.collections.clear()

        max_width = max(tx_square_side, rx_square_side) / 2
        ax.set_zlim((max_width * -2, max_width * 2))
        ax.set_ylim((max_width * -2, max_width * 2))
        ax.set_xlim((max_width * -2, max_width * 2))
        ax.plot((0, 0), (0, 0), (-max_width + 1, max_width - 1), 'b', label='z-axis')
        ax.plot((-max_width + 1, max_width - 1), (0, 0), (0, 0), 'r', label='x-axis')
        ax.plot((0, 0), (-max_width + 1, max_width - 1), (0, 0), 'k', label='y-axis')
        plt.axis('off')

        plot_towers()
        v_vec = ax.quiver(tx[0], tx[1] + Radius, tx[2], 0, 1, 0,
                          length=2500, normalize=True, fc='k', ec='k')
        v_ann.set_position((tx[0] - 0.5e3, tx[1] + 1.5e3 + Radius))

        plot_tx(radius=Radius)
        cur_time.set_text('t = {:.2E} s'.format(t))

        for u in range(num_towers):
            print('Tower {}: t = {}, rec_times[u] = {}'.format(u, t, rec_times[u]))
            if t >= rec_times[u]:
                tower_text[u].set_text('Tower {} received at t = {} s'.format(u, rec_times[u]))

    def animate2(i):
        global t0_rec, t1_rec, t2_rec, t3_rec, TDOA1, TDOA2, TDOA3, TDOA4, TDOA5, TDOA6, v_vec, v_ann, TDOA1_dist, \
            TDOA2_dist, TDOA3_dist, TDOA4_dist, TDOA5_dist, TDOA6_dist

        ax.clear()
        ax.collections.clear()

        max_width = max(tx_square_side, rx_square_side) / 2
        ax.set_zlim((max_width * -2, max_width * 2))
        ax.set_ylim((max_width * -2, max_width * 2))
        ax.set_xlim((max_width * -2, max_width * 2))
        ax.plot((0, 0), (0, 0), (-max_width + 1, max_width - 1), 'b', label='z-axis')
        ax.plot((-max_width + 1, max_width - 1), (0, 0), (0, 0), 'r', label='x-axis')
        ax.plot((0, 0), (-max_width + 1, max_width - 1), (0, 0), 'k', label='y-axis')
        plt.axis('off')

        plot_lines()
        plot_towers()

        d = i / n_frames * max_d

        first_tower = int(np.argmin(rec_times))
        d0 = [np.clip(d, d, distances[first_tower])]
        kugel_1 = Kugeln(radius=d0, x=x0[first_tower], y=y0[first_tower], z=z0[first_tower])
        kugel_1.coordinaten()


        for j in [x for x in range(towers.shape[0]) if x != first_tower]:
            TDOA_j = v * (rec_times[j] - rec_times[first_tower])
            d1 = [np.clip(d, d + TDOA_j, distances[j])]
            print('tower', str(first_tower), 'to', str(j))
            kugel_i = Kugeln(radius=d1, x=x0[j], y=y0[j], z=z0[j])
            kugel_i.coordinaten()

            #d = i / n_frames * max_d
#
            #first_tower = int(np.argmin(rec_times))
            #circles[first_tower].radius = d
#
            #for j in [x for x in range(towers.shape[0]) if x != first_tower]:
            #    # print('tower', str(first_tower), 'to', str(j))
            #    locus = get_locus(tower_1=(towers[first_tower][0],
            #                               towers[first_tower][1]),
            #                      tower_2=(towers[j][0], towers[j][1]),
            #                      time_1=rec_times[first_tower],
            #                      time_2=rec_times[j],
            #                      v=v, delta_d=delta_d, max_d=d)
            #    locus_plots[j].set_xdata(locus[0])
            #    locus_plots[j].set_ydata(locus[1])
#
            #    TDOA_j = v * (rec_times[j] - rec_times[first_tower])
            #    circles[j].radius = d + TDOA_j
        r1 = np.clip(d, d + TDOA1_dist, r[0])
        r2 = np.clip(d, d + TDOA2_dist, r[1])
        r3 = np.clip(d, d + TDOA3_dist, r[2])
        r4 = np.clip(d, d + TDOA4_dist, r[0])
        r5 = np.clip(d, d + TDOA5_dist, r[1])
        r6 = np.clip(d, d + TDOA6_dist, r[2])
        #circle12(radius=r1)
        #circle13(radius=r2)
        #circle14(radius=r3)
        #circle23(radius=r4)
        #circle24(radius=r5)
        #circle34(radius=r6)
        #print("\r d = : " + str(d), end="\n")
        #print("\r Tower 0 radius r0 = : " + str(d0), end="\n")
        #print("\r Tower 1 radius r1 = : " + str(d1 + TDOA_dist1), end="\n")
        #print("\r Tower 2 radius r2 = : " + str(d2 + TDOA_dist2), end="\n")
        #print("\r Tower 3 radius r3 = : " + str(d3 + TDOA_dist3), end="\n")


    anim_1 = FuncAnimation(fig, animate1, frames=n_frames, interval=1, blit=False, repeat=False)
    # anim.save('C:/Users/Mem/Desktop/Studium/Vertiefungsmodul/Animationen/TDOA.gif', writer='imagemagick', fps=60)
    # plt.close()
    # Image(url='C:/Users/Mem/Desktop/Studium/Vertiefungsmodul/Animationen/TDOA.gif')
    plt.show()

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(projection='3d')
    fig.subplots_adjust(left=0.2, right=0.8)
    plt.axis('off')

    anim_2 = FuncAnimation(fig, animate2, frames=n_frames, interval=16, blit=False, repeat=True)
    # anim.save('C:/Users/Mem/Desktop/Studium/Vertiefungsmodul/Animationen/TDOA.gif', writer='imagemagick', fps=60)
    #anim_2.save('/home/mohammed/Animationen/TDOA.gif', writer='imagemagick', fps=60)
    #plt.close()
    # Image(url='C:/Users/Mem/Desktop/Studium/Vertiefungsmodul/Animationen/TDOA.gif')
    plt.show()



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
