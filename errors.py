import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import sympy as sy
from trilateration import Trilateration_3D
from scipy.optimize import curve_fit

#num_towers = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
num_towers = []
num = 60
num_towers += [i for i in range(4, num, 1)]
print(num_towers)
tx_square_side = 5e3
rx_square_side = 30e3
v = 299792458
t_0 = 2.5
delta_d = int(100)
max_d = int(20e3)
rec_time_noise_stdd = 1e-3

error_sy, error_tri = [], []
for x in num_towers:
    towers_0 = (np.random.rand(x, 3) - 0.5) * rx_square_side
    array_2dd = np.array(towers_0)
    zahl_0 = np.repeat(1, x)
    array_1d_0 = np.array(zahl_0)
    towers_1 = array_2dd * array_1d_0[:, np.newaxis]
    zahl = [1, 1, 0]
    array_2d = np.array(towers_1)
    array_1d = np.array(zahl)
    towers = array_2d * array_1d[np.newaxis, :]

    tx = (np.random.rand(3) - [0.5, 0.5, -1]) * tx_square_side

    decimal_p = 9
    formatted_values_tx = [("{:.{}f}".format(x, decimal_p)) for x in tx]
    formatted_string_tx = ", ".join(formatted_values_tx)

    distances = np.array([np.sqrt((x[0] - tx[0]) ** 2 + (x[1] - tx[1]) ** 2 + (x[2] - tx[2]) ** 2)
                          for x in towers])
    distances += np.random.normal(loc=0, scale=rec_time_noise_stdd,
                                  size=x)

    rec_times = distances / v
    # Add noise to receive times
    rec_times += np.random.normal(loc=0, scale=rec_time_noise_stdd,
                                  size=x)

    # coordinates of the towers and their radii.
    x0, y0, z0 = [], [], []
    for i in range(towers.shape[0]):
        x0.__iadd__([towers[i][0]])
        y0.__iadd__([towers[i][1]])
        z0.__iadd__([towers[i][2]])

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
        decimal_places = 9
        initial_value = 50
        initial_solution = (initial_value, initial_value, initial_value)
        # Solve the system of equations for x, y and z coordinates
        solutions = sy.nsolve(system, (x, y, z), initial_solution, maxsteps=50, verify=False, rational=True,
                              prec=decimal_places)


        positions = Trilateration_3D(towers, distances)

        positions_array = np.array(positions)
        # Check if z coordinate is negative and make it positive
        if (positions_array[:, 2] < 0).any():
                positions_array[:, 2] = np.abs(positions_array[:, 2])
        def format_positions(posi, decimal_places):
            formatted_values = [("[{}]".format(", ".join(["{:.{}f}".format(x, decimal_places) for x in pos.tolist()])))
                                for pos in posi]
            return formatted_values
        formatted_positions = format_positions(positions_array, decimal_places=12)
        mean_position = np.mean(positions_array, axis=0)


        # Calculate the average error
        original_locations = np.array(tx)
        sy_locations = np.array(solutions)
        sy_locations = sy_locations.reshape(mean_position.shape)
        absolute_difference_sy = np.abs(original_locations - sy_locations)
        absolute_difference_tri = np.abs(original_locations - mean_position)
        average_error_sy = np.mean(absolute_difference_sy)
        average_error_tri = np.mean(absolute_difference_tri)
        #print("Average error sy.nsolve:", average_error_sy)
        #print("Average error Trilatation:", average_error_tri)
        return [average_error_sy, average_error_tri]
    solveEquations()
    errors = solveEquations()
    error_sy.append(errors[0])
    error_tri.append(errors[1])

def func(x, a, b, c):
    return a * np.exp(-b * x) + c
def func_tri(x, a, b, c):
    return a * x**(-b) + c

popt, pcov = curve_fit(func, num_towers, error_sy)
popt_tri, pcov_tri = curve_fit(func_tri, num_towers, error_tri)

# Generate a finer grid of x values for the plot
x_fit = np.linspace(min(num_towers), max(num_towers), num)

# Evaluate the fitted function with the optimized parameters
y_fit = func(x_fit, *popt)
y_fit_tri = func_tri(x_fit, *popt_tri)

# Plot the original data and the fitted curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
#ax.plot(num_towers, error_sy, label='error_sy')
##ax.plot(num_towers, error_tri, label='error_tri')
ax1.plot(x_fit, y_fit, label='Fitted Curve sy')
ax2.plot(x_fit, y_fit_tri, label='Fitted Curve tri')
ax1.legend()
ax2.legend()
plt.show()


