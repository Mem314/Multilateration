import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import sympy as sy
import math
from trilateration import sphereCircle, Trilateration_3D

# How many towers. All towers receive the transmission.
num_towers = 4

# Metre length of a square containing the transmitting
# device, centred around (x, y) = (0, 0). Device will be randomly placed
# in this area.
tx_square_side = 5e3

# Metre length of a square containing the towers,
# centred around (x, y) = (0, 0). towers will be randomly placed
# in this area.
rx_square_side =25e3

# Speed of transmission propogation. Generally equal to speed of
# light for radio signals.
v = 299792458

# Time at which transmission is performed. Really just useful to
# make sure the code is using relative times rather than depending on one
# of the receive times being zero.
t_0 = 2.5

# Metre increments to radii of circles when generating locus of
# circle intersection.
delta_d = int(10)

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
plot_trilateration_spheres = True

# Whether to plot a straight line
# between every pair of towers. This is useful for visualising the
# hyperbolic loci focal points.
plot_lines_between_towers = False
plot_lines_to_tx = True


# Generate towers with x and y coordinates.
# for tower i: x, y = towers[i][0], towers[i][1]
towersss = (np.random.rand(num_towers, 3)-0.5) * rx_square_side
array_2dd = np.array(towersss)
zahl1 = [1,1,1,1]
array_1dd = np.array(zahl1)
towerss = array_2dd * array_1dd[:,np.newaxis]
zahl = [1,1,0]
array_2d = np.array(towerss)
array_1d = np.array(zahl)
towers = array_2d * array_1d[np.newaxis,:]
print('towers:\n', towers)

# location of transmitting device with tx[0] being x and tx[1] being y.
tx = (np.random.rand(3)-[0.5, 0.5, -1]) * tx_square_side

print('tx:', tx)

# Distances from each tower to the transmitting device,
# simply triangle hypotenuse.
# distances[i] is distance from tower i to transmitter.
distances = np.array([((x[0]-tx[0])**2 + (x[1]-tx[1])**2 + (x[2]-tx[2])**2)**0.5
                       for x in towers])
print('distances:', distances)

# Time at which each tower receives the transmission.
rec_times = distances/v + t_0
# Add noise to receive times
rec_times += np.random.normal(loc=0, scale=rec_time_noise_stdd,
                              size=num_towers)
print('rec_times:', rec_times)

# Plot towers and transmission location.
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
max_width = max(tx_square_side*2, rx_square_side*2)/2
ax.set_zlim((max_width * -1, max_width))
ax.set_ylim((max_width * -1, max_width))
ax.set_xlim((max_width * -1, max_width))
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

for i in range(towers.shape[0]):
    x = towers[i][0]
    y = towers[i][1]
    z = towers[i][2]
    ax.scatter3D(x, y, z)
    ax.text3D(x, y, z,'Tower '+str(i) )
    ax.scatter3D(tx[0], tx[1], tx[2])
    ax.text3D(tx[0], tx[1], tx[2], 'Tx')

# Iterate over every unique combination of towers and plot nifty stuff.
for i in range(num_towers):
    if(plot_lines_to_tx):
        #line between transmitter and tx
        ax.plot3D((towers[i][0], tx[0]),
                (towers[i][1], tx[1]),
                (towers[i][2], tx[2]))
    if(plot_trilateration_spheres):
        # Kugeln
        Theta, Phi = np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 40)
        theta1, phi = np.meshgrid(Theta, Phi)
        X1, Y1, Z1 = [], [], []
        for i in range(0, len(towers)):
            X1.__iadd__([x0[i] + r[i] * np.sin(phi) * np.cos(theta1)])
            Y1.__iadd__([y0[i] + r[i] * np.sin(phi) * np.sin(theta1)])
            Z1.__iadd__([z0[i] + r[i] * np.cos(phi)])
            plot_sphere = ax.plot_surface(
                X1[i], Y1[i], Z1[i], rstride=4, cstride=4, cmap=cm.coolwarm,
                linewidth=0, antialiased=False, alpha=0.05)
    for j in range(i+1, num_towers):
        if(plot_lines_between_towers):
            # Line between towers
            ax.plot3D((towers[i][0], towers[j][0]),
                      (towers[i][1], towers[j][1]),
                      (towers[i][2], towers[j][2]))


d = x0[1]-x0[0]
g = y0[1]-y0[0]
h = z0[1]-z0[0]
e = x0[2]-x0[0]
f = y0[2]-y0[0]
n = z0[2]-z0[0]
a = x0[3]-x0[0]
b = y0[3]-y0[0]
c = z0[3]-z0[0]

#d, e, f = sy.symbols("d e f")
x, y, z = sy.symbols("x y z")
equations = [
    sy.Eq((x - x0[0]) ** 2 + (y - y0[0]) ** 2 + (z - z0[0]) ** 2, r[0] ** 2),
    sy.Eq((x - d - x0[0]) ** 2 + (y - g - y0[0]) ** 2 + (z - h - z0[0]) ** 2, r[1] ** 2),
    sy.Eq((x - e - x0[0]) ** 2 + (y - f - y0[0]) ** 2 + (z - n - z0[0]) ** 2, r[2] ** 2),
    sy.Eq((x - a - x0[0]) ** 2 + (y - b - y0[0]) ** 2 + (z - c - z0[0]) ** 2, r[3] ** 2),
]

if np.sqrt(d ** 2 + g ** 2 + h ** 2) <= (r[0]+r[1]) and \
        np.sqrt(e ** 2 + f ** 2 + n ** 2) <= (r[0] + r[2]) and \
        np.sqrt(a ** 2 + b ** 2 + c ** 2) <= (r[0] + r[3]) and \
        np.sqrt((abs(d) - abs(e))**2 + (abs(g)-abs(f))**2 + (abs(h)-abs(n))**2) <= (r[1] + r[2]) and \
        np.sqrt((abs(d) - abs(a))**2 + (abs(g)-abs(b))**2 + (abs(h)-abs(c))**2) <= (r[1] + r[3]) and \
        np.sqrt((abs(e) - abs(a))**2 + (abs(f)-abs(b))**2 + (abs(n)-abs(c))**2) <= (r[2] + r[3]):
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
        #eqPoint1 = ax.scatter3D(posi[0], posi[1], posi[2])
        #eqPoint2 = eqPoint1
else:
    location = print(f'there is no intesection')
    eqPoint1 = [0, 0, 0]
    eqPoint2 = [0, 0, 0]

theta, te = np.linspace(0, 2 * np.pi, 80), np.linspace(0, 2 * np.pi, 80)
if np.sqrt(d ** 2 + g ** 2 + h ** 2) <= (r[0]+r[1]):
    X12 = (d ** 2 + g ** 2 + h ** 2 + r[0]**2 - r[1]**2) / (2*np.sqrt(d ** 2 + g ** 2 + h ** 2)*r[0])
    coord= sphereCircle(r[0], 0, math.atan2(g, d), np.arccos(X12), theta)
    array_2d = np.array(coord)
    Zahl = [x0[0], y0[0], z0[0]]
    array_1d = np.array(Zahl)
    coordinats_12 = array_2d + array_1d[:, np.newaxis]
    plot_circle_1 = ax.plot(
       coordinats_12[0], coordinats_12[1], coordinats_12[2])
else:
    plot_circle_1 = []

if np.sqrt(e ** 2 + f ** 2 + n ** 2) <= (r[0] + r[2]):
    X13 = (r[0]**2 - r[2]**2 + e**2 + f**2 + n**2)/(2*r[0]*np.sqrt(e**2 + f**2 + n**2))
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

if np.sqrt((abs(d) - abs(e))**2 + (abs(g)-abs(f))**2 + (abs(h)-abs(n))**2) <= (r[1] + r[2]):
    X23 = 2*np.sqrt((d-e) ** 2 + (f-g)**2 + (n-h)**2)*r[1] / ((d - e)**2 + (f-g)**2 + (n-h)**2 + r[1]**2 - r[2]**2)
    coord = sphereCircle(r[1], np.pi -math.atan2(n-h, np.sqrt((d-e)**2 + (g-f)**2)), (math.atan2(g-f, d-e)),
                         asec(X23), te)
    array_2d = np.array(coord)
    Zahl = [d+x0[0], g+y0[0], h+z0[0]]
    array_1d = np.array(Zahl)
    coordinats_23 = array_2d + array_1d[:, np.newaxis]
    plot_circle_3 = ax.plot(
        coordinats_23[0], coordinats_23[1], coordinats_23[2])
else:
    plot_circle_3 = []

if np.sqrt(a ** 2 + b ** 2 + c ** 2) <= (r[0] + r[3]):
    X14 = (r[0]**2 - r[3]**2 + a**2 + b**2 + c**2) / (2*r[0]*np.sqrt(a**2 + b**2 + c**2))
    coord = sphereCircle(r[0],  math.atan2(c, np.sqrt(a**2 + b**2)), math.atan2(b, a), np.arccos(X14), te)
    array_2d = np.array(coord)
    Zahl = [x0[0], y0[0], z0[0]]
    array_1d = np.array(Zahl)
    coordinats_14 = array_2d + array_1d[:, np.newaxis]
    plot_circle_4 = ax.plot(
        coordinats_14[0], coordinats_14[1], coordinats_14[2])
else:
    plot_circle_4 = []

if np.sqrt((abs(d) - abs(a))**2 + (abs(g)-abs(b))**2 + (abs(h)-abs(c))**2) <= (r[1] + r[3]):
    X24 = 2*np.sqrt((d - a)**2 + (b-g)**2 + (c-h)**2)*r[1] / ((d - a)**2 + (g-b)**2 + (h-c)**2 + r[1]**2 - r[3]**2)
    coord = sphereCircle(r[1], np.pi -math.atan2(c-h, np.sqrt((d-a)**2 + (g-b)**2)), (math.atan2(g-b, d-a)),
                         asec(X24), te)
    array_2d = np.array(coord)
    Zahl = [d+x0[0], g+y0[0], h+z0[0]]
    array_1d = np.array(Zahl)
    coordinats_24 = array_2d + array_1d[:, np.newaxis]
    plot_circle_5 = ax.plot(
        coordinats_24[0], coordinats_24[1], coordinats_24[2])
else:
    plot_circle_5 = []

if np.sqrt((abs(e) - abs(a))**2 + (abs(f)-abs(b))**2 + (abs(n)-abs(c))**2) <= (r[2] + r[3]):
    X34 = 2*np.sqrt((e-a)**2 + (f-b)**2 + (c-n)**2)*r[2] / ((e-a)**2 + (f-b)**2 + (c-n)**2 + r[2]**2 - r[3]**2)
    coord = sphereCircle(r[2],np.pi - math.atan2(c-n, np.sqrt((e-a)**2 + (f-b)**2)), (math.atan2(f-b, e-a)),
                         asec(X34), te)
    array_2d = np.array(coord)
    Zahl = [e+x0[0], f+y0[0], n+z0[0]]
    array_1d = np.array(Zahl)
    coordinats_34 = array_2d + array_1d[:, np.newaxis]
    plot_circle_6 = ax.plot(
        coordinats_34[0], coordinats_34[1], coordinats_34[2])
else:
    plot_circle_6 = []


plt.show()

