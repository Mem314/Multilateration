import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import sympy as sy
from scipy.optimize import curve_fit



def Trilateration_3D(towers, distances):
    """
    This program implements trilateration in 3 dimensions.
    It aims to determine the position of a point (posi) in space based on the
    position (coordinates) of P_i other points in space and the distance
    from the unknown point to these known points.

    At least 4 spheres are required to determine posi. The intersection of the first two spheres is a circle.
    The intersection of two circles is a geodesic. Therefore, at least 4 spheres are required.
    By using the intersections, we obtain 3 circles and by using their intersections,
    we obtain 2 geodesics. The intersection of the geodesics gives us the position of posi.
    For 4 spheres, we obtain 1 geodesic ==> for (4+n) spheres, we obtain (n+1) geodesics.

    This implementation is a Cartesian 'true-range multilateration'.
    Using the first point and the Pythagorean theorem, we can calculate the distances
    between posi and the centers of the spheres.
    """

    positions = []
    num_towers = len(towers)

    for i in range(num_towers - 3):
        print("i =" , i)
        # Select a subset of 4 towers
        towers_subset = towers[i:i + 4]
        distances_subset = distances[i:i + 4]

        # coordinates: p1 = [x, y, z] and radii.
        p = []
        for j in range(len(towers_subset)):
            p.append(np.array(towers_subset[j][:3], dtype=np.float128))
        r = np.array(distances_subset, dtype=np.float128)

        # the unit vector in the direction from p1 to p2.
        d, i_vals, j_vals, e_x, e_y, e_z = [], [], [], [], [], []
        for k in range(len(towers_subset) - 3):
            d.append(np.linalg.norm(p[k + 1] - p[k]).astype(np.float128))
            # projection of the vector from p3 to p1 onto e_x.
            e_x.append((p[k + 1] - p[k]) / np.linalg.norm(p[k + 1] - p[k]).astype(np.float128))
            i_vals.append(np.dot(e_x[k], (p[k + 2] - p[k])).astype(np.float128))
            e_y.append((p[k + 2] - p[k] - (i_vals[k] * e_x[k])) / np.linalg.norm(
                p[k + 2] - p[k] - (i_vals[k] * e_x[k])).astype(np.float128))
            j_vals.append(np.dot(e_y[k], (p[k + 2] - p[k])).astype(np.float128))
            e_z.append(np.cross(e_x[k], e_y[k]).astype(np.float128))

        x, y, z1, z2 = [], [], [], []
        for k in range(len(towers_subset) - 3):
            x_val = ((r[k] ** 2) - (r[k + 1] ** 2) + (d[k] ** 2)) / (2 * d[k])
            x.append(x_val.astype(np.float128))
            y_val = (((r[k] ** 2) - (r[k + 2] ** 2) + (i_vals[k] ** 2) + (j_vals[k] ** 2)) / (2 * j_vals[k])) - (
                        (i_vals[k] / j_vals[k]) * x_val)
            y.append(y_val.astype(np.float128))
            z1.append(np.sqrt(np.abs(r[k] ** 2 - x_val ** 2 - y_val ** 2)).astype(np.float128))
            z2.append((z1[k] * (-1)).astype(np.float128))

        ans1, ans2, dist1, dist2 = [], [], [], []
        for k in range(len(towers_subset) - 3):
            ans1.append((p[k] + (x[k] * e_x[k]) + (y[k] * e_y[k]) + (z1[k] * e_z[k])).astype(np.float128))
            print("ans1: ", ans1)
            ans2.append((p[k] + (x[k] * e_x[k]) + (y[k] * e_y[k]) + (z2[k] * e_z[k])).astype(np.float128))
            dist1.append(np.linalg.norm(p[k + 3] - ans1[k]).astype(np.float128))
            dist2.append(np.linalg.norm(p[k + 3] - ans2[k]).astype(np.float128))


        mean_position = np.mean(ans1, axis=0, dtype=np.float128)

        # Append the calculated position to the list of positions
        positions.append(mean_position.astype(np.float128))
    return positions


if __name__ == "__main__":
    num = 7
    num_towers = [i for i in range(4, num+1, 1)]
    print(num_towers)

    rx_square_side = 1
    v = 299792458
    rec_time_noise_stdd = 0
    precision = 12

    tx = (np.random.rand(3).astype(np.float128) - [0.5, 0.5, -1]) * np.float128(rx_square_side)
    formatted_values_tx = [("{:.{}f}".format(x, precision)) for x in tx]
    formatted_string_tx = ", ".join(formatted_values_tx)
    print("The locations of tx is:", formatted_string_tx)
    towers_0 = (np.random.rand(max(num_towers), 3).astype(np.float128) - 0.5) * np.float128(rx_square_side)
    towers = towers_0 * np.array([1, 1, 0], dtype=np.float128)


    for u in num_towers:
        # Use the sliced towers within the u loop
        towers_u = towers[:u]

        distances = np.array([np.sqrt((x[0] - tx[0]) ** 2 + (x[1] - tx[1]) ** 2 + (x[2] - tx[2]) ** 2)
                              for x in towers_u], dtype=np.float128)
        distances += np.random.normal(loc=0, scale=rec_time_noise_stdd,
                                      size=u)
        # Print out the data
        print("The input points, in the format of [x, y, z], are:")
        for i, (tower, distance) in enumerate(zip(towers_u, distances)):
            print(f"Tower {i + 1}: {tower} Distance: {distance}")

        positions = Trilateration_3D(towers_u, distances)

        positions_array = np.array(positions, dtype=np.float128)
        # Check if z coordinate is negative and if so, make it positive
        if (positions_array[:, 2] < 0).any():
            positions_array[:, 2] = np.abs(positions_array[:, 2])

        def format_positions(posi, decimal_places):
            formatted_values = [
                ("[{}]".format(", ".join(["{:.{}f}".format(np.float128(x), decimal_places) for x in pos.tolist()])))
                for pos in posi
            ]
            return formatted_values
        formatted_positions = format_positions(positions_array, decimal_places=precision)
        original_locations = np.array(tx, dtype=np.float128)

        for pos in formatted_positions:
            print("Position: {}".format(pos))
        mean_position = np.mean(positions_array, axis=0, dtype=np.float128)
        print("mean of the positions: {}".format(mean_position))

        mean_error_list = []

        for i, position in enumerate(formatted_positions):
            absolute_difference_tri = np.abs(positions_array - original_locations, dtype=np.float128)
            mean_error_tri = np.mean(absolute_difference_tri).astype(np.float128)
            mean_error_list.append((mean_error_tri).astype(np.float128))
            print("Position {}: Mean error to tx: {}".format(i + 1, mean_error_tri))

        mean_error_array = np.array(mean_error_list).astype(np.float128)
        print("mean_error_array: {}".format(mean_error_array))


    def linear_model(x, a, b):
        return a * x + b


    # Fit the data using the custom exponential model with weights
    params_tri, _ = curve_fit(linear_model, num_towers, mean_error_array, method='trf')

    # Generate x-values for the plot
    x = np.linspace(min(num_towers), max(num_towers), 80)

    # Compute the fitted curve using the optimized parameters
    fit_curve_tri = linear_model(x, params_tri[0], params_tri[1])

    # Plot the original data and the fitted curve
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.scatter(num_towers, mean_error_array, label='error_tri')
    ax.plot(x, fit_curve_tri, color='blue', label='Fitted Curve (Tri)')
    plt.xlabel('Number of Towers')
    plt.ylabel('Error')
    ax.legend()
    ax.set_yscale('asinh')
    ylim = 1e-11
    ax.set_ylim(bottom=-ylim*2, top=ylim*2)

    # Add text annotation for parameter 'a'
    text_x = max(num_towers) * 0.965  # x-coordinate for the text annotation
    text_y = ylim * 2 * 1.05  # y-coordinate for the text annotation
    text = f'a = {params_tri[0]:}'  # text annotation with parameter 'a' value
    ax.text(text_x, text_y, text, fontsize=12, ha='center')

    plt.show()
