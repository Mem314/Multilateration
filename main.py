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

# Speed of transmission propagation. Generally equal to speed of
# light for radio signals.
v = 299792458

# Time at which transmission is performed. Really just useful to
# make sure the code is using relative times rather than depending on one
# of the reception times being zero.
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

def dfasda():
    pass

x0, y0, z0 = [], [], []
for i in range(towers.shape[0]):
    x0.__iadd__([towers[i][0]])
    y0.__iadd__([towers[i][1]])
    z0.__iadd__([towers[i][2]])

r = []
for i in range(towers.shape[0]):
    r.__iadd__([distances[i]])

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(projection='3d')
fig.subplots_adjust(left=0.2, right=0.8)
max_width = max(tx_square_side, rx_square_side) / 2
ax.set_zlim((max_width * -2, max_width * 2))
ax.set_ylim((max_width * -2, max_width * 2))
ax.set_xlim((max_width * -2, max_width * 2))
ax.axis('off')
ax.plot((0, 0), (0, 0), (-max_width + 1, max_width - 1), 'b', label='z-axis')
ax.plot((-max_width + 1, max_width - 1), (0, 0), (0, 0), 'r', label='x-axis')
ax.plot((0, 0), (-max_width + 1, max_width - 1), (0, 0), 'k', label='y-axis')


if plot_trilateration_spheresIntersection_circles:

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
                X, Y, Z, rstride=2, cstride=2, cmap=cm.cividis,
                linewidth=0, antialiased=False, alpha=0.2)
            return plot

    def solveEquations():
        x, y, z = sy.symbols("x y z")
        first_tower = int(np.argmin(rec_times))
        dx, dy, dz = [], [], []
        disti = []
        equations = []
        eq0 = sy.Eq((x - x0[first_tower]) ** 2 + (y - y0[first_tower]) ** 2 +
                    (z - z0[first_tower]) ** 2, distances[first_tower] ** 2)
        equations.append(eq0)
        for i in [x for x in range(towers.shape[0]) if x != first_tower]:
            dx.append(x0[i] - x0[first_tower])
            dy.append(y0[i] - y0[first_tower])
            dz.append(z0[i] - z0[first_tower])
            disti.append(distances[i])
        for i in range(towers.shape[0] - 1):
            eq = sy.Eq((x - dx[i] - x0[first_tower]) ** 2 + (y - dy[i] - y0[first_tower]) ** 2 +
                       (z - dz[i] - z0[first_tower]) ** 2, disti[i] ** 2)
            equations.append(eq)
        # Convert the list of equations into a list of Expr objects
        exprs = [eq.lhs - eq.rhs for eq in equations]

        # Create the system of equations as a Sympy Matrix object
        system = sy.Matrix(exprs)

        # set the initial solution for the numerical method
        initial_solution = (50, 50, 50)
        # Solve the system of equations for x, y and z coordinates
        solutions = sy.nsolve(system, (x, y, z), initial_solution, maxsteps=50, verify=False, rational=True)

        print(f'sy.nsolve locations is: {solutions}')
        posi = Trilateration_3D(towers, distances)
        if posi[2] < 0:
            posi[2] = -posi[2]
        print(f'Trilateration_3D location is: {posi}')
    solveEquations()

    def circles(radius_0, radius):

        theta, te = np.linspace(0, 2 * np.pi, 80), np.linspace(0, 2 * np.pi, 80)
        first_tower = int(np.argmin(rec_times))
        dx, dy, dz = [], [], []
        disti = []
        dx.insert(first_tower, x0[first_tower])
        dy.insert(first_tower, y0[first_tower])
        dz.insert(first_tower, z0[first_tower])
        disti.insert(first_tower, distances[first_tower])
        for i in [x for x in range(towers.shape[0]) if x != first_tower]:
            dx.insert(i, x0[i] - x0[first_tower])
            dy.insert(i, y0[i] - y0[first_tower])
            dz.insert(i, z0[i] - z0[first_tower])
            disti.insert(i, distances[i])

        def asec(x):
            if x < -1 or x > 1:
                return math.acos(1 / x)
            else:
                return 0

        for i in [x for x in range(towers.shape[0]) if x != first_tower]:
            if np.sqrt(dx[i]**2 + dy[i]**2 + dz[i]**2) <= (disti[first_tower] + disti[i]):
                X_0 = (dx[i]**2 + dy[i]**2 + dz[i]**2 + disti[first_tower]**2 - disti[i]**2) /(
                      2*disti[first_tower]*np.sqrt((dx[i]**2 + dy[i]**2 + dz[i]**2)))
                coord_0 = sphereCircle(radius_0, 0, math.atan2(dy[i], dx[i]), np.arccos(X_0), theta)
                #coord_0 = sphereCircle(radius_0, math.atan2(dz[i], np.sqrt(dy[i] ** 2 + dx[i] ** 2)),
                #                       math.atan2(dy[i], dx[i]), np.arccos(X_0), theta)
                array_2d = np.array(coord_0)
                Zahl = [dx[first_tower], dy[first_tower], dz[first_tower]]
                array_1d = np.array(Zahl)
                coordinats_0 = array_2d + array_1d[:, np.newaxis]
                #plot_circle_0 = ax.plot(
                #    coordinats_0[0], coordinats_0[1], coordinats_0[2], color='g')

            for j in [y for y in range(i, towers.shape[0]) if y != first_tower and y != i]:
                if np.sqrt((dx[j] - dx[i]) ** 2 + (dy[j] - dy[i]) ** 2 + (dz[j] - dz[i]) ** 2) <= (disti[j] + disti[i]):
                    X = 2 * np.sqrt(((dx[j] - dx[i]) ** 2 + (dy[j] - dy[i]) ** 2 +
                                     (dz[j] - dz[i]) ** 2)) * disti[i] / \
                        ((dx[j] - dx[i]) ** 2 + (dy[j] - dy[i]) ** 2 +
                         (dz[j] - dz[i]) ** 2 + disti[j] ** 2 - disti[i] ** 2)
                    coord = sphereCircle(radius, np.pi -
                                         math.atan2(dz[i] - dz[first_tower], np.sqrt(
                                             (dx[first_tower] - dx[i]) ** 2 + (dy[first_tower] - dy[i]) ** 2)),
                                         (math.atan2(dy[first_tower] - dy[i], dx[first_tower] - dx[i])),
                                         asec(X), te)
                    array_2d = np.array(coord)
                    Zahl = [dx[i] + dx[j], dy[i] + dy[j], dz[i] + dz[j]]
                    array_1d = np.array(Zahl)
                    coordinats = array_2d + array_1d[:, np.newaxis]
                    plot_circle = ax.plot(
                        coordinats[0], coordinats[1], coordinats[2], color='b')
                else:
                    plot_circle = []
            else:
                plot_circle_0 = []



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

    v_vec = ax.quiver(tx[0] + 1e3, tx[1], tx[2], 1, 0, 0,
                      length=10000, normalize=False, fc='k', ec='k')
    v_ann = ax.text3D(tx[0] + 4e3, tx[1] + 4e3, tx[2], 'v = {} m/s'.format(v))
    t_rec = 0

    TDOA = []
    TDOA_dist = []
    for i in range(num_towers):
        for j in range(i + 1, num_towers):
            tdoa = abs(rec_times[j] - rec_times[i])
            TDOA.append(tdoa)
            TDOA_dist.append(v * tdoa)

    n_frames = 10
    max_seconds = 1e-4
    max_d = 1e5

    def animate1(i):
        global t_rec, tower_text, v_vec, v_ann

        t = i / n_frames * max_seconds
        Radius = v * t

        ax.collections.clear()
        plot_towers()
        v_vec = ax.quiver(tx[0] + Radius, tx[1], tx[2], 1, 0, 0,
                          length=10000, normalize=True, fc='k', ec='k')
        v_ann.set_position((tx[0] + 4e3 + Radius, tx[1] + 4e3))

        kugel_tx = Kugeln(radius=Radius, x=tx[0], y=tx[1], z=tx[2])
        kugel_tx.coordinaten()
        cur_time.set_text('t = {:.2E} s'.format(t))

        for u in range(num_towers):
            print('Tower {}: t = {}, rec_times[u] = {}'.format(u, t, rec_times[u]))
            if t >= rec_times[u]:
                tower_text[u].set_text('Tower {} received at t = {} s'.format(u, rec_times[u]))


    def animate2(i):
        global TDOA, TDOA_dist

        ax.clear()
        ax.collections.clear()

        max_width = max(tx_square_side, rx_square_side) / 2
        ax.set_zlim((max_width * -2, max_width * 2))
        ax.set_ylim((max_width * -2, max_width * 2))
        ax.set_xlim((max_width * -2, max_width * 2))
        ax.plot((0, 0), (0, 0), (-max_width + 1, max_width - 1), 'b', label='z-axis')
        ax.plot((-max_width + 1, max_width - 1), (0, 0), (0, 0), 'r', label='x-axis')
        ax.plot((0, 0), (-max_width + 1, max_width - 1), (0, 0), 'k', label='y-axis')
        ax.axis('off')

        plot_lines()
        plot_towers()


        d = i / n_frames * max_d
        first_tower = int(np.argmin(rec_times))
        d0 = [np.clip(d, d, distances[first_tower])]
        circles(radius_0=d0, radius=d)
        kugel_0 = Kugeln(radius=d0, x=x0[first_tower], y=y0[first_tower], z=z0[first_tower])
        kugel_0.coordinaten()
        for j in [x for x in range(towers.shape[0]) if x != first_tower]:
            TDOA_j = v * (rec_times[j] - rec_times[first_tower])
            d1 = [np.clip(d, d + TDOA_j, distances[j])]
            kugel_i = Kugeln(radius=d1, x=x0[j], y=y0[j], z=z0[j])
            kugel_i.coordinaten()

            for i in [y for y in range(towers.shape[0]) if y != first_tower and y != j]:
                d2 = [np.clip(d, d, distances[i])]
                circles(radius_0=0, radius=d2)



            #print("\r d = : " + str(d), end="\n")
            #print("\r Tower 0 radius r0 = : " + str(d0), end="\n")
            #print("\r Tower 1 radius r1 = : " + str(d1 + TDOA_dist1), end="\n")
            #print("\r Tower 2 radius r2 = : " + str(d2 + TDOA_dist2), end="\n")
            #print("\r Tower 3 radius r3 = : " + str(d3 + TDOA_dist3), end="\n")


    anim_1 = FuncAnimation(fig, animate1, frames=n_frames, interval=1, blit=False, repeat=False)
    ## anim.save('C:/Users/Mem/Desktop/Studium/Vertiefungsmodul/Animationen/TDOA.gif', writer='imagemagick', fps=60)
    ## plt.close()
    ## Image(url='C:/Users/Mem/Desktop/Studium/Vertiefungsmodul/Animationen/TDOA.gif')
    plt.show()

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(projection='3d')
    fig.subplots_adjust(left=0.2, right=0.8)
    ax.axis('off')


    anim_2 = FuncAnimation(fig, animate2, frames=n_frames, interval=10, blit=False, repeat=False)
    # anim.save('C:/Users/Mem/Desktop/Studium/Vertiefungsmodul/Animationen/TDOA.gif', writer='imagemagick', fps=60)
    # anim_2.save('/home/mohammed/Animationen/TDOA.gif', writer='imagemagick', fps=60)
    # plt.close()
    # Image(url='C:/Users/Mem/Desktop/Studium/Vertiefungsmodul/Animationen/TDOA.gif')
    plt.show()

"""
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
"""
