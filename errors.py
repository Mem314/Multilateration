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
num = 190
num_towers = [i for i in range(4, num+1, 1)]
print(num_towers)

field_area = 1500  # Area of the field in square meters
rx_square_side = np.sqrt(field_area)  # Length of the side of a square field
v = 299792458
receive_time_noise = 1e-9
precision = 12  # max precision 15, because tx can only be presented with 15 float-point
np.set_printoptions(precision=precision)

tx = (np.random.rand(3).astype(np.longdouble) - [0.5, 0.5, -1]) * np.sqrt(field_area)
formatted_values_tx = [("{:.{}f}".format(x, precision)) for x in tx]
formatted_string_tx = ", ".join(formatted_values_tx)
print("The locations of tx is:", formatted_string_tx)



towers_0 = (np.random.rand(max(num_towers), 3).astype(np.longdouble) - 0.5) * np.longdouble(rx_square_side)
towers = towers_0 * np.array([1, 1, 0], dtype=np.longdouble)

error_sy = []
for x in num_towers:
    towers_x = towers[:x]

    distances = np.array([np.sqrt((x[0] - tx[0]) ** 2 + (x[1] - tx[1]) ** 2 + (x[2] - tx[2]) ** 2)
                          for x in towers_x], dtype=np.longdouble)
    distances += np.random.normal(loc=0, scale=receive_time_noise,
                                 size=x)

    rec_times = distances / v
    # Add noise to receive times
    rec_times += np.random.normal(loc=0, scale=receive_time_noise,
                                  size=x)

    # coordinates of the towers and their radii.
    x0, y0, z0 = [], [], []

    # Set decimal precision
    decimal.getcontext().prec = precision

    for i in range(towers_x.shape[0]):
        x0.append(decimal.Decimal(str(towers_x[i][0])))
        y0.append(decimal.Decimal(str(towers_x[i][1])))
        z0.append(decimal.Decimal(str(towers_x[i][2])))


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
        for i in [x for x in range(towers_x.shape[0]) if x != first_tower]:
            dx.append(x0[i] - x0[first_tower])
            dy.append(y0[i] - y0[first_tower])
            dz.append(z0[i] - z0[first_tower])
            disti.append(distances[i])
        for i in range(towers_x.shape[0] - 1):
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

        # Calculate the average error
        original_locations = np.array(tx)
        solutions_array = np.array(solutions)
        reshaped_original_locations = np.reshape(original_locations, (3, 1))
        absolute_difference_sy = np.abs(solutions_array - reshaped_original_locations)
        average_error_sy = np.mean(absolute_difference_sy)
        #print("sy_locations: {}".format(reshaped_original_locations))
        #print("absolute_difference_sy: {}".format(absolute_difference_sy))

        return [average_error_sy]

    error_sy.append(solveEquations(precision=precision)[0])
print(error_sy)
#error_sy = [7.84873735737897279e-12, 1.23962108662436140e-11, 3.58914906620442601e-11, 2.39841266783345312e-11,
#            8.90926503578637341e-12, 1.46429921862636000e-11, 9.84571517197163626e-12, 6.30595742001175778e-12,
#            7.67019947267115014e-12, 2.26659265719723428e-12, 4.69191186192504291e-12, 2.24018669644591532e-12,
#            8.90926503578637341e-12, 8.57969417444407837e-12, 1.22176729815357913e-11, 7.24240755619702063e-12,
#            7.39399000649250867e-12, 6.33236338076307674e-12, 7.36703457208017406e-12, 6.60912232060273386e-12,
#            8.30293523460442125e-12, 6.24843052010141490e-12, 5.11970377839917242e-12, 8.17830821872126782e-12,
#            5.39646271823882954e-12]

def exponential_model(x, a, b, c):
    return a * np.exp(-b * x) + c


# Convert the error lists to numpy arrays
error_sy = np.array(error_sy)

# Fit the data using the custom exponential model with weights
#params_sy, _ = curve_fit(exponential_model, num_towers, error_sy, maxfev=10000, method='trf', loss='huber')
params_sy, _ = curve_fit(exponential_model, num_towers, error_sy, maxfev=10000, method='lm', ftol=1e-10, xtol=1e-12)


# Generate x-values for the plot
x = np.linspace(min(num_towers), max(num_towers), 100)

# Compute the fitted curve using the optimized parameters
fit_curve_sy = exponential_model(x, params_sy[0], params_sy[1], params_sy[2])

# Plot the original data and the fitted curve
fig, ax = plt.subplots(figsize=(12, 8))
ax.grid(which='major', color='#DDDDDD', linewidth=0.8, zorder=5)
ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5, zorder=5)
ax.minorticks_on()
ax.scatter(num_towers, error_sy, label='error_sy', s=10, c='k', marker='o', zorder=10)
ax.plot(x, fit_curve_sy, color='red', label='Fitted Curve (SY)')
plt.xlabel('Number of Towers')
plt.ylabel('Error')
#ylim2 = 5e-11 # Set the desired y-axis limits for the fitted curve
#ax.set_ylim(bottom=-ylim2, top=ylim2)
ax.legend()
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
