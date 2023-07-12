from mpl_toolkits.mplot3d import *
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import sympy as sy
import scipy
import math
import sys
import time
from trilateration import Trilateration_3D
from matplotlib.animation import FuncAnimation
from IPython.display import Image
from IPython.display import display, clear_output
from typing import Any
from dataclasses import dataclass

"""
This program implements multitrilateration in 3 dimensions.
It aims to determine the position of a point (posi) in space based on the
position (coordinates) of P points in space and the distance
from the unknown point to these known points.

At least 4 spheres are required to determine the position. The intersection of the first two spheres is a circle.
The intersection of two circles is a geodesic. Therefore, at least 4 spheres are required.
By using the intersections, we obtain 3 circles and by using their intersections,
we obtain 2 geodesics. The intersection of the geodesics gives us the position of posi.
For 4 spheres, we obtain 1 geodesic ==> for (4+n) spheres, we obtain (n+1) geodesics.
"""


num_towers = 4
field_area = 1500  # Area of the field in square meters
rx_square_side = np.sqrt(field_area)  # Length of the side of a square field
v = 299792458
receive_time_noise = 1e-12
precision = 12

plot_trilateration_tx = True
plot_trilateration_spheres = True
plot_trilateration_spheresIntersection_circles = True

plot_lines_between_towers = False
plot_lines_to_tx = True

# The Towers
towers_0 = (np.random.rand(num_towers, 3).astype(np.longdouble) - 0.5) * np.sqrt(field_area)
towers = towers_0 * np.array([1, 1, 0], dtype=np.longdouble)
print("Towers:", towers)

# location of transmitting device (Sender).
tx = (np.random.rand(3).astype(np.longdouble) - [0.5, 0.5, -1]) * np.sqrt(field_area)
formatted_values_tx = [("{:.{}f}".format(x, precision)) for x in tx]
formatted_string_tx = ", ".join(formatted_values_tx)
print("The locations of tx is:", formatted_string_tx)

# Distances from each tower to the transmitting device,
# simply triangle hypotenuse.
# distances[i] is distance from tower i to transmitter.
distances = np.array([np.sqrt((x[0] - tx[0]) ** 2 + (x[1] - tx[1]) ** 2 + (x[2] - tx[2]) ** 2)
                              for x in towers], dtype=np.longdouble)
distances += np.random.normal(loc=0, scale=receive_time_noise,
                                    size=num_towers)
print('distances:', distances)

# Time at which each tower receives the transmission.
rec_times = distances / v
# Add noise to receive times
rec_times += np.random.normal(loc=0, scale=receive_time_noise,
                              size=num_towers)
print('rec_times:', rec_times)

"""
Create a visualisation for TDOA.
"""

# coordinates of the towers and their radii.
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
max_width = rx_square_side / 2
ax.set_zlim((max_width * -3, max_width * 3))
ax.set_ylim((max_width * -3, max_width * 3))
ax.set_xlim((max_width * -3, max_width * 3))
ax.axis('off')
plt.tight_layout(pad=0.05)
ax.plot3D((0, 0), (0, 0), (-max_width * 5, max_width * 5), 'b--', label='z-axis')
ax.plot3D((-max_width * 5, max_width * 5), (0, 0), (0, 0), 'r--', label='x-axis')
ax.plot3D((0, 0), (-max_width * 5, max_width * 5), (0, 0), 'k--', label='y-axis')
ax.legend()

if plot_trilateration_spheresIntersection_circles:

    def plot_towers():
        for k in range(towers.shape[0]):
            x = towers[k][0]
            y = towers[k][1]
            z = towers[k][2]
            ax.text3D(x + 2, y + 2, z , 'Tower ' + str(k))
            ax.text3D(tx[0] + 2, tx[1] + 2, tx[2] + 2, 'Tx')
            ax.scatter3D(x, y, z, color="b", s=20)
            ax.scatter3D(tx[0], tx[1], tx[2], color="k", s=30)


    def plot_lines():
        for i in range(num_towers):
            if plot_lines_to_tx:
                # arrow between transmitter and tx
                ax.quiver(tx[0], tx[1], tx[2], towers[i][0] - tx[0], towers[i][1] - tx[1], towers[i][2] - tx[2],
                          arrow_length_ratio=0.1)
            for j in range(i + 1, num_towers):
                if plot_lines_between_towers:
                    # Line between towers
                    pl2 = ax.plot3D((towers[i][0], towers[j][0]),
                              (towers[i][1], towers[j][1]),
                              (towers[i][2], towers[j][2]), color='b')


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
                X, Y, Z, rstride=1, cstride=5, cmap=cm.winter,
                linewidth=0, antialiased=False, alpha=0.15)
            return plot


    def sphereCircle(r, psi, zeta, alpha, t):
        """
        3D circle on a sphere with  r
        :param r: radius of the Circle
        :param psi:  rotation around y-axis
        :param zeta: rotation around z-axis
        :param alpha: angular radius of the circle
        :param t: parameter (0-2*Pi)
        :return:
        """
        x = r * (np.cos(alpha) * np.cos(zeta) * np.cos(psi) +
                 np.sin(alpha) * (-np.sin(t) * np.sin(zeta) + np.cos(t) * np.cos(zeta) * np.sin(psi)))
        y = r * (np.cos(zeta) * np.sin(alpha) * np.sin(t) +
                 np.sin(zeta) * (np.cos(alpha) * np.cos(psi) + np.cos(t) * np.sin(alpha) * np.sin(psi)))
        z = r * (-np.cos(t) * np.cos(psi) * np.sin(alpha) + np.cos(alpha) * np.sin(psi))
        return [x, y, z]


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
        initial_value = 50
        initial_solution = (initial_value, initial_value, initial_value)
        # Solve the system of equations for x, y and z coordinates
        solutions = sy.nsolve(system, (x, y, z), initial_solution, maxsteps=50, verify=False, rational=True,
                              prec=precision)


        formatted_values_sy = [("{:.{}f}".format(x, precision)) for x in solutions]
        formatted_string_sy = ", ".join(formatted_values_sy)
        print("sy.nsolve locations is:", formatted_string_sy)

        positions = Trilateration_3D(towers, distances)

        positions_array = np.array(positions)
        formatted_array = np.around(positions_array, decimals=12)
        # Check if z coordinate is negative and make it positive
        if (positions_array[:, 2] < 0).any():
            positions_array[:, 2] = np.abs(positions_array[:, 2])
        array_string = np.array2string(formatted_array, precision=12, separator=', ')
        print("position_array: {}".format(array_string))


        def format_positions(posi, decimal_places):
            formatted_values = [("[{}]".format(", ".join(["{:.{}f}".format(x, decimal_places) for x in pos.tolist()])))
                                for pos in posi]
            return formatted_values
        formatted_positions = format_positions(positions_array, decimal_places=precision)
        for pos in formatted_positions:
            print("Position: {}".format(pos))
        mean_position = np.mean(positions_array, axis=0, dtype=np.longdouble)
        print("mean of the positions: {}".format(mean_position))

        # Calculate the average error
        original_locations = np.array(tx)
        sy_locations = np.array(solutions)
        sy_locations = sy_locations.reshape(mean_position.shape)
        absolute_difference_sy = np.abs(original_locations - sy_locations)
        absolute_difference_tri = np.abs(original_locations - mean_position)
        average_error_sy = np.mean(absolute_difference_sy)
        average_error_tri = np.mean(absolute_difference_tri)
        print("Average error sy.nsolve:", average_error_sy)
        print("Average error Trilatation:", average_error_tri)
        return [average_error_sy, average_error_tri]
    #solveEquations()


    def solveEquations_Linearisation():
        print("----------------------------------------------------- Linearisation -----------")

        """
        This vignette illustrates the ideas behind solving systems of linear equations of the form Ax=b where:
        - A is an m×n matrix of coefficients for m equations in n unknowns
        - x is an n×1 vector unknowns, x1,x2,…xn
        - b is an m×1 vector of constants, the “right-hand sides” of the equations
        
        The general conditions for solutions are:
        - the equations are consistent (solutions exist) if r(A|b)=r(A)
            - the solution is unique if r(A|b)=r(A)=n
            - the solution is underdetermined if r(A|b)=r(A)<n
        - the equations are inconsistent (no solutions) if r(A|b)>r(A)
        """
        x, y, z = sy.symbols("x y z")

        k = []
        for i in range(towers.shape[0]):
            k.append(x0[i]**2 + y0[i]**2 + z0[i]**2)

        """
        Moreover, vector b  consists of  distances between the  unknown point N 
        and  all  the  reference  points.  Especially  in  static  sensor 
        networks  the  computation  of  the  entire  localization  can  be 
        accomplished in the nodes themselves, since the computation 
        is restricted to a matrix vector  multiplication of matrix L and 
        vector b. 
         """
        b = []
        for i in range(1, towers.shape[0]):
            eq_b = (distances[0]**2 - distances[i]**2 - k[0] + k[i])
            b.append(eq_b)
        b = sy.Matrix(b)

        A = sy.Matrix([])
        for i in range(1, towers.shape[0]):
            row = sy.Matrix([[x0[i] - x0[0], y0[i] - y0[0], z0[i] - z0[0]]])
            A = A.row_insert(i, row)

        A = 2 * A
        r = sy.Matrix([[x], [y], [z]])

        """
        solution_1 is using r=A^(-1)*b to solve the problem, which is not possible duo to the A^(-1).
        This is because we set the Towers to have z=0, the det of A is then not defined and with that
        the inverse of A is not defined. With this method only x and y coordinates are calculated.
        """
        solution_1 = sy.solve(A * r - b, r)
        print("solution_1:", solution_1)

        """
        Also here is the same Problem duo to the invers of (A^T * T).
        """
        A_T = A.T
        #eqution_2 = sy.Eq((A_T*A).inv() * A_T * b, r)
        #solution_2 = sy.solve(eqution_2, r)
        #print("solution_2:", solution_2)

        equation_3 = sy.Eq(A*r, b)
        solution_3 = sy.solve(equation_3, r)
        print("solution_3:", solution_3)

        equation_4 = sy.Eq((A_T * A)*r, A_T * b)
        solution_4 = sy.solve(equation_4, r)
        print("solution_4:", solution_4)

        """
         When the matrix A is singular (z=0), we get either no solution or infinite solution to the problem A*r=b, 
         thats why we consider the pseudoinverse approach and that may yield inaccurate results, especially if the 
         singular values are close to zero. In such cases, the system of equations is ill-conditioned, and finding an 
         accurate solution becomes challenging.
        """
        # Compute the pseudoinverse of A
        A_pseudoinv = A.pinv()

        # Compute the solution vector r using the pseudoinverse
        r = np.dot(A_pseudoinv, b)

        # Print the solution vector r
        print("Solution vector r:")
        print(r)
    #solveEquations_Linearisation()


    def circles(radius_0, radius):
        def asec(x):
            if x < -1 or x > 1:
                return math.acos(1 / x)
            else:
                return 0

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
                plot_circle_0 = ax.plot(
                    coordinats_0[0], coordinats_0[1], coordinats_0[2], color='r')

            for j in [y for y in range(i, towers.shape[0]) if y != first_tower and y != i]:
                if np.sqrt((dx[j] - dx[i]) ** 2 + (dy[j] - dy[i]) ** 2 + (dz[j] - dz[i]) ** 2) <= (disti[j] + disti[i]):
                    X = 2 * np.sqrt(((dx[j] - dx[i]) ** 2 + (dy[j] - dy[i]) ** 2 +
                                     (dz[j] - dz[i]) ** 2)) * disti[i] / \
                        ((dx[i] - dx[j]) ** 2 + (dy[i] - dy[j]) ** 2 +
                         (dz[i] - dz[j]) ** 2 + disti[i] ** 2 - disti[j] ** 2)
                    radius_1 = np.clip(radius, radius, disti[i])
                    coord = sphereCircle(radius_1,
                                         math.atan2(dz[j] - dz[i], np.sqrt(
                                             (dx[j] - dx[i]) ** 2 + (dy[j] - dy[i]) ** 2)),
                                         (math.atan2(dy[j] - dy[i], dx[j] - dx[i])),
                                         asec(X), te)
                    array_2d = np.array(coord)
                    Zahl = [dx[i] + dx[first_tower], dy[i] + dy[first_tower], dz[i] + dz[first_tower]]
                    array_1d = np.array(Zahl)
                    coordinats = array_2d + array_1d[:, np.newaxis]
                    plot_circle = ax.plot(
                        coordinats[0], coordinats[1], coordinats[2], color='r')
                else:
                    plot_circle = []
            else:
                plot_circle_0 = []



    # Annotations, to be updated during animation
    l = rx_square_side
    k = 5
    cur_time = ax.text(40, l - 4*k , k , 't = 0')
    tower_text = []
    for i in range(num_towers):
        text = ax.text(40, l - 4*k, -i * k , 'Tower {} received at t = '.format(i))
        tower_text.append(text)

    v_vec = ax.quiver(tx[0] + 1, tx[1], tx[2], 1, 0, 0,
                      length=10, normalize=False, fc='k', ec='k')
    v_ann = ax.text3D(tx[0] + 4, tx[1] + 4, tx[2], 'v = {} m/s'.format(v))

    t_rec = 0
    TDOA = []
    TDOA_dist = []
    for i in range(num_towers):
        for j in range(i + 1, num_towers):
            tdoa = abs(rec_times[j] - rec_times[i])
            TDOA.append(tdoa)
            TDOA_dist.append(v * tdoa)

    n_frames =120
    max_seconds = 1.1 * max(rec_times)
    max_d = 4 * max(distances)

    def animate1(i):
        global t_rec, tower_text, v_vec, v_ann

        t = i / n_frames * max_seconds
        Radius = v * t

        ax.collections.clear()
        plot_towers()
        plot_lines()

        v_vec = ax.quiver(tx[0] + Radius, tx[1], tx[2], 1, 0, 0,
                          length=10, normalize=True, fc='k', ec='k')
        v_ann.set_position((tx[0] + 4 + Radius, tx[1] + 4))

        kugel_tx = Kugeln(radius=Radius, x=tx[0], y=tx[1], z=tx[2])
        kugel_tx.coordinaten()
        cur_time.set_text('t = {:.12E} s'.format(t))

        for u in range(num_towers):
            print('Tower {}: t = {}, rec_times[{}] = {}'.format(u, t, u, rec_times[u]))
            if t >= rec_times[u]:
                tower_text[u].set_text('Tower {} received at t = {} s'.format(u, rec_times[u]))


    def animate2(i):
        global TDOA, TDOA_dist

        ax.clear()
        ax.collections.clear()

        max_width = rx_square_side / 2
        ax.set_zlim((max_width * -3, max_width * 3))
        ax.set_ylim((max_width * -3, max_width * 3))
        ax.set_xlim((max_width * -3, max_width * 3))
        ax.plot3D((0, 0), (0, 0), (-max_width * 5, max_width * 5), 'b--', label='z-axis')
        ax.plot3D((-max_width * 5, max_width * 5), (0, 0), (0, 0), 'r--', label='x-axis')
        ax.plot3D((0, 0), (-max_width * 5, max_width * 5), (0, 0), 'k--', label='y-axis')
        ax.legend()
        ax.axis('off')

        ax.view_init(elev=40, azim=i/2)

        plot_lines()
        plot_towers()

        d = i / n_frames * max_d
        first_tower = int(np.argmin(rec_times))
        last_tower = int(np.argmax(rec_times))
        TDOA_0 = v * (rec_times[last_tower] - rec_times[first_tower])
        TDOA_2 = v * (rec_times[last_tower] - rec_times[first_tower])
        d0 = [np.clip(d, d - TDOA_0, distances[first_tower])]
        d2 = [np.clip(d, d + TDOA_2, distances[last_tower])]


        circles(radius_0=d0, radius=d)
        kugel_0 = Kugeln(radius=d0, x=x0[first_tower], y=y0[first_tower], z=z0[first_tower])
        kugel_0.coordinaten()
        for j in [x for x in range(towers.shape[0]) if x != first_tower]:
            TDOA_j = v * (rec_times[j] - rec_times[first_tower])
            d1 = [np.clip(d, d + TDOA_j, distances[j])]
            kugel_i = Kugeln(radius=d1, x=x0[j], y=y0[j], z=z0[j])
            kugel_i.coordinaten()



            print("\r d = : " + str(d), end="\n")
            #print("\r Tower 0 radius r0 = : " + str(d0), end="\n")
            #print("\r Tower 1 radius r1 = : " + str(d1 + TDOA_dist1), end="\n")
            #print("\r Tower 2 radius r2 = : " + str(d2 + TDOA_dist2), end="\n")
            #print("\r Tower 3 radius r3 = : " + str(d3 + TDOA_dist3), end="\n")



    anim_1 = FuncAnimation(fig, animate1, frames=n_frames, interval=1, blit=False, repeat=False)
    ## anim.save('C:/Users/Mem/Desktop/Studium/Vertiefungsmodul/Animationen/TDOA.gif', writer='imagemagick', fps=60)
    #anim_1.save('/home/mohammed/Animationen/TDOA1.gif', writer='imagemagick', fps=20)
    plt.show()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection='3d')
    fig.tight_layout()
    ax.axis('off')


    anim_2 = FuncAnimation(fig, animate2, frames=n_frames, interval=10, blit=False, repeat=False)
    #anim_2.save('C:/Users/Mem/Desktop/Studium/Vertiefungsmodul/Animationen/TDOA.gif', fps=12)
    #anim_2.save('/home/mohammed/Animationen/TDOA2.gif', writer='imagemagick', fps=15)
    plt.show()

"""
https://rosap.ntl.bts.gov/view/dot/12134  ==> TIME OF ARRIVAL EQUATIONS (wichtig fürs schreiben)
https://www.th-luebeck.de/fileadmin/media_cosa/Dateien/Veroeffentlichungen/Sammlung/TR-2-2015-least-sqaures-with-ToA.pdf
https://www.tandfonline.com/doi/epdf/10.1080/00107518108231543?needAccess=true&role=button (paper für cosmic shower)
"""

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
