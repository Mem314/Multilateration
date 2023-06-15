import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import sympy as sy
from trilateration import Trilateration_3D
from scipy.optimize import curve_fit, leastsq, minimize
from sklearn.linear_model import RANSACRegressor, HuberRegressor
from scipy.interpolate import UnivariateSpline
from sklearn.base import BaseEstimator
import decimal
from mpmath import mp

num_towers, num_towers_0, num_towers_1 = [], [], []
num = 120
num_towers_0 += [i for i in range(4, int(num/7), 1)]
num_towers_1 += [i for i in range(int(num/7), num, 10)]
num_towers = num_towers_0 + num_towers_1
print(num_towers)
tx_square_side = 5e3
rx_square_side = 30e3
v = 299792458
rec_time_noise_stdd = 10e-9

precision = 9

tx = (np.random.rand(3).astype(np.longdouble) - [0.5, 0.5, -1]) * np.longdouble(rx_square_side)
formatted_values_tx = [("{:.{}f}".format(x, precision)) for x in tx]
formatted_string_tx = ", ".join(formatted_values_tx)
print("The locations of tx is:", formatted_string_tx)

error_sy, error_tri = [], []
for x in num_towers:
    towers_0 = (np.random.rand(x, 3).astype(np.longdouble) - 0.5) * np.longdouble(rx_square_side)
    towers = towers_0 * np.array([1, 1, 0], dtype=np.longdouble)

    distances = np.array([np.sqrt((x[0] - tx[0]) ** 2 + (x[1] - tx[1]) ** 2 + (x[2] - tx[2]) ** 2)
                          for x in towers], dtype=np.longdouble)
    #distances += np.random.normal(loc=0, scale=rec_time_noise_stdd,
    #                              size=x)

    rec_times = distances / v
    # Add noise to receive times
    #rec_times += np.random.normal(loc=0, scale=rec_time_noise_stdd,
    #                              size=x)

    # coordinates of the towers and their radii.
    x0, y0, z0 = [], [], []

    # Set decimal precision
    decimal.getcontext().prec = precision

    for i in range(towers.shape[0]):
        x0.append(decimal.Decimal(str(towers[i][0])))
        y0.append(decimal.Decimal(str(towers[i][1])))
        z0.append(decimal.Decimal(str(towers[i][2])))


    def solveEquations(precision):
        mp.dps = precision  # Set the precision for mpmath

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
        initial_value = mp.mpf(50)
        initial_solution = (initial_value, initial_value, initial_value)
        # Solve the system of equations for x, y, and z coordinates
        solutions = sy.nsolve(system, (x, y, z), initial_solution, maxsteps=10000, verify=False, rational=False,
                              prec=precision)

        positions = Trilateration_3D(towers, distances)

        positions_array = np.array(positions)
        # Check if z coordinate is negative and if so, make it positive
        if (positions_array[:, 2] < 0).any():
            positions_array[:, 2] = np.abs(positions_array[:, 2])

        def format_positions(posi, decimal_places):
            formatted_values = [("[{}]".format(", ".join(["{:.{}f}".format(x, decimal_places) for x in pos.tolist()])))
                                for pos in posi]
            return formatted_values

        formatted_positions = format_positions(positions_array, decimal_places=precision)
        mean_position = np.mean(positions_array, axis=0)

        # Calculate the average error
        original_locations = np.array(tx)
        sy_locations = np.array(solutions)
        sy_locations = sy_locations.reshape(mean_position.shape)
        absolute_difference_sy = np.abs(original_locations - sy_locations)
        absolute_difference_tri = np.abs(original_locations - mean_position)
        average_error_sy = np.mean(absolute_difference_sy)
        average_error_tri = np.mean(absolute_difference_tri)

        return [average_error_sy, average_error_tri]


    errors = solveEquations(precision=precision)
    error_sy.append(errors[0])
    error_tri.append(errors[1])


def exponential_model(x, a, b, c):
    return a * np.power(x, b) + c
def linear_model(x, a, b):
    return a * x + b

# Convert the error lists to numpy arrays
error_sy = np.array(error_sy)
error_tri = np.array(error_tri)

# Fit the data using the custom exponential model with weights
params_sy, _ = curve_fit(exponential_model, num_towers, error_sy, method='trf', loss='cauchy')
params_tri, _ = curve_fit(linear_model, num_towers, error_tri, method='trf', loss='arctan')

# Generate x-values for the plot
x = np.linspace(min(num_towers), max(num_towers), 80)

# Compute the fitted curve using the optimized parameters
fit_curve_sy = exponential_model(x, params_sy[0], params_sy[1], params_sy[2])
fit_curve_tri = linear_model(x, params_tri[0], params_tri[1])


# Plot the original data and the fitted curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
ax1.scatter(num_towers, error_sy, label='error_sy')
ax2.scatter(num_towers, error_tri, label='error_tri')
ax1.plot(x, fit_curve_sy, color='red', label='Fitted Curve (SY)')
ax2.plot(x, fit_curve_tri, color='blue', label='Fitted Curve (Tri)')
plt.xlabel('Number of Towers')
plt.ylabel('Error')
ax1.legend()
ax2.legend()
plt.show()

"""
# Fit the data using spline interpolation
spline_sy = UnivariateSpline(num_towers, error_sy, s=0.1, k=1)
spline_tri = UnivariateSpline(num_towers, error_tri, s=0.1, k=1)

# Generate x-values for the plot
x = np.linspace(min(num_towers), max(num_towers), 100)

# Compute the fitted curves
fit_curve_sy = spline_sy(x)
fit_curve_tri = spline_tri(x)
"""


"""
# Define the matrix A and vector b for the least squares problem
A_sy = np.column_stack((np.ones_like(num_towers), np.exp(-num_towers)))
b_sy = error_sy

A_tri = np.column_stack((np.ones_like(num_towers), np.exp(-num_towers)))
b_tri = error_tri

# Solve the least squares problem
x_sy, _, _, _ = np.linalg.lstsq(A_sy, b_sy, rcond=None)
x_tri, _, _, _ = np.linalg.lstsq(A_tri, b_tri, rcond=None)

# Generate x-values for the plot
x = np.linspace(min(num_towers), max(num_towers), 400)

# Compute the fitted curves
fit_curve_sy = x_sy[0] + x_sy[1] * np.exp(-x)
fit_curve_tri = x_tri[0] + x_tri[1] * np.exp(-x)
"""


"""
coefficients_sy = np.polyfit(num_towers, error_sy, 8)
fit_curve_sy = np.poly1d(coefficients_sy)

coefficients_tri = np.polyfit(num_towers, error_tri, 8)
fit_curve_tri = np.poly1d(coefficients_tri)

# Generate x-values for the plot
x = np.linspace(min(num_towers), max(num_towers), 400)
"""


"""
def func(x, a, b, c):
    return a * np.exp(-b * x) + c
def func_tri(x, a, b, c):
    return a * np.exp(-b * x) + c

popt, pcov = curve_fit(func, num_towers, error_sy, maxfev=1800)
popt_tri, pcov_tri = curve_fit(func_tri, num_towers, error_tri, method='lm' ,maxfev=800)

# Generate a finer grid of x values for the plot
x_fit = np.linspace(min(num_towers), max(num_towers), num)

# Evaluate the fitted function with the optimized parameters
y_fit = func(x_fit, *popt)
y_fit_tri = func_tri(x_fit, *popt_tri)
"""
