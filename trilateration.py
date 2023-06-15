import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpmath import mp


def Trilateration_3D(towers, distances):
    """
    This program implements trilateration in 3 dimensions.
    It aims to determine the position of a point (posi) in space based on the
    position (coordinates) of P points in space and the distance
    from the unknown point to these known points.

    This implementation is a Cartesian 'true-range multilateration'.
    By using the first point and the Pythagorean theorem, we can calculate the distances
    between posi and the centers of the spheres.
    """

    positions = []
    num_towers = len(towers)

    for i in range(num_towers - 3):
        # Select a subset of 4 towers
        towers_subset = towers[i:i + 4]
        distances_subset = distances[i:i + 4]

        # coordinates: p1 = [x, y, z] and radii.
        p = []
        for j in range(len(towers_subset)):
            p.append(np.array(towers_subset[j][:3], dtype=np.float128))
        r = np.array(distances_subset, dtype=np.float128)

        d, i_vals, j_vals, e_x, e_y, e_z = [], [], [], [], [], []
        for k in range(len(towers_subset) - 3):
            # the Euclidean distance between p[k + 1] and p[k]
            d.append(np.linalg.norm(p[k + 1] - p[k]).astype(np.float128))
            # the unit vector e_x in the direction from p1 to p2.
            e_x.append((p[k + 1] - p[k]) / np.linalg.norm(p[k + 1] - p[k]).astype(np.float128))
            # the projection of the vector from p[k + 2] to p[k] onto the e_x vector.
            i_vals.append(np.dot(e_x[k], (p[k + 2] - p[k])).astype(np.float128))
            #  the unit vector e_y: subtracting the component of the vector from p[k] to p[k + 2] that lies along e_x
            e_y.append((p[k + 2] - p[k] - (i_vals[k] * e_x[k])) / np.linalg.norm(
                p[k + 2] - p[k] - (i_vals[k] * e_x[k])).astype(np.float128))
            # the projection of the vector from p[k + 2] to p[k] onto the e_y vector.
            j_vals.append(np.dot(e_y[k], (p[k + 2] - p[k])).astype(np.float128))
            # the cross product of the e_x and e_y vectors, resulting in the orthogonal unit vector e_z.
            e_z.append(np.cross(e_x[k], e_y[k]).astype(np.float128))

        x, y, z1, z2 = [], [], [], []
        for k in range(len(towers_subset) - 3):
            x_val = ((r[k] ** 2) - (r[k + 1] ** 2) + (d[k] ** 2)) / (2 * d[k])
            x.append(x_val.astype(np.float128))
            # calculates the x-coordinate of the trilaterated point based on the distances and radii.
            y_val = (((r[k] ** 2) - (r[k + 2] ** 2) + (i_vals[k] ** 2) + (j_vals[k] ** 2)) / (2 * j_vals[k])) - (
                        (i_vals[k] / j_vals[k]) * x_val)
            y.append(y_val.astype(np.float128))
            # calculates the y-coordinate of the trilaterated point based on the distances,radii, and projection values.
            z1.append(np.sqrt(np.abs(r[k] ** 2 - x_val ** 2 - y_val ** 2)).astype(np.float128))
            # calculates the positive z-coordinate of the trilaterated point based on the distances, radii, and the x
            # and y coordinates. It uses the Pythagorean theorem to find the z-coordinate.
            z2.append((z1[k] * (-1)).astype(np.float128))
            # calculates the negative z-coordinate by multiplying the positive z-coordinate z1 with (-1), because
            # trilateration can have two possible solutions in 3D space, and
            # the negative value represents the other solution.

        ans1, ans2, dist1, dist2 = [], [], [], []
        for k in range(len(towers_subset) - 3):
            # the first possible trilaterated point is calculated by summing the reference point p[k] with the
            # respective components along the x, y, and z axes. The values x[k], y[k], and z1[k] are multiplied by their
            # respective unit vectors e_x[k], e_y[k], and e_z[k], and the results are added together.
            ans1.append((p[k] + (x[k] * e_x[k]) + (y[k] * e_y[k]) + (z1[k] * e_z[k])).astype(np.float128))
            ans2.append((p[k] + (x[k] * e_x[k]) + (y[k] * e_y[k]) + (z2[k] * e_z[k])).astype(np.float128))
            # the distance between the four Point and the calculated one from ans1 and ans2.
            dist1.append(np.linalg.norm(p[k + 3] - ans1[k]).astype(np.float128))
            dist2.append(np.linalg.norm(p[k + 3] - ans2[k]).astype(np.float128))

        mean_positions = np.mean(ans1, axis=0, dtype=np.float128)

        # Append the calculated position to the list of positions
        positions.append(mean_positions.astype(np.float128))
    return positions


if __name__ == "__main__":
    num = 20
    num_towers = [i for i in range(4, num+1, 1)]
    print(num_towers)

    field_area = 1500  # Area of the field in square meters
    rx_square_side = np.sqrt(field_area)  # Length of the side of a square field
    v = 299792458
    receive_time_noise = 1e-15
    precision = 15  # max precision 15, because tx can only be presented with 15 float-point

    tx = (np.random.rand(3).astype(np.float128) - [0.5, 0.5, -1]) * np.sqrt(field_area)
    formatted_values_tx = [("{:.{}f}".format(x, precision)) for x in tx]
    formatted_string_tx = ", ".join(formatted_values_tx)
    print("The locations of tx is:", formatted_string_tx)

    towers_0 = (np.random.rand(max(num_towers), 3).astype(np.float128) - 0.5) * np.sqrt(field_area)
    towers = towers_0 * np.array([1, 1, 0], dtype=np.float128)

    locations = []

    for u in num_towers:
        # Use the sliced towers within the u loop
        towers_u = towers[:u]

        distances = np.array([np.sqrt((x[0] - tx[0]) ** 2 + (x[1] - tx[1]) ** 2 + (x[2] - tx[2]) ** 2)
                              for x in towers_u], dtype=np.float128)
        distances += np.random.normal(loc=0, scale=receive_time_noise,
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

        original_location = np.array(tx)  # .astype(str).astype(mp.mpf)
        print("original_locations_mpf: {}".format(tx))
        formatted_positions = []
        for pos in positions_array:
            pos_formatted = [np.format_float_scientific(x, unique=False, precision=precision) for x in pos]
            formatted_positions.append(pos_formatted)

        mean_position = np.mean(positions_array, axis=0, dtype=np.float128)
        mean_position_formatted = np.array2string(mean_position, precision=precision,
                                                  formatter={'float_kind': lambda x: "{:.15e}".format(x)})

        print("Formatted Positions:")
        for pos_formatted in formatted_positions:
            print("Position: {}".format(pos_formatted))
        print("Mean of the positions: {}".format(mean_position_formatted))
        locations.append(mean_position_formatted)

        mean_position_mpf = np.array(mean_position, dtype=np.float128)
        original_locations_mpf = np.array(original_location, dtype=np.float128)
        print("original_locations_mpf: {}".format(original_locations_mpf))

        mean_error_list = []

        for i, position in enumerate(formatted_positions):
            mp.dps = precision

            current_position_mpf = np.array(position, dtype=np.float128)

            result_mpf = np.abs(np.subtract(current_position_mpf, tx, dtype=np.float128))
            absolute_difference_tri = np.array(result_mpf, dtype=np.float128)
            np.set_printoptions(precision=precision)

            print("absolute_difference_tri:", absolute_difference_tri)

            mean_error_tri = np.mean(absolute_difference_tri).astype(np.float128)

            mean_error_formatted = np.format_float_scientific(mean_error_tri, unique=False, precision=precision)
            mean_error_list.append(mean_error_formatted)
            print("Position {}: Mean error to tx: {}".format(i + 1, mean_error_tri))

        mean_error_array = np.array(mean_error_list).astype(np.float128)
        mean_error_array_formatted = [np.format_float_scientific(elem, unique=False, precision=precision) for elem in
                                      mean_error_array]


    absolute_mean_array_error = mean_error_array_formatted
    absolute_mean_array_error_numeric = np.array(absolute_mean_array_error, dtype=np.float128)
    print("absolute_mean_array_error: {}".format(absolute_mean_array_error_numeric))

    print(positions_array)
    tx_array = np.array(tx)

    tx_proj = tx_array[:2]

    # Convert Cartesian coordinates to polar coordinates
    r = np.linalg.norm(tx_proj)
    theta_polar = np.arctan2(tx_proj[1], tx_proj[0])

    # Create polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Plot data points in polar coordinates
    ax.scatter(positions_array, np.ones_like(positions_array), label='error_tri', s=30, c='b', marker='o')

    # Plot 3D point in polar coordinates
    #ax.scatter(theta_polar, r, label='original', s=100, c='r', marker='o')
    ax.scatter(0, 0, label='original', s=100, c='r', marker='o')

    # Set grid lines
    ax.grid(True)

    # Set legend
    ax.legend()

    """
    Fit Kurve
    """

    def linear_model(x, a, b):
        return a * x + b

    # Fit the data using the custom exponential model with weights
    params_tri, _ = curve_fit(linear_model, num_towers, absolute_mean_array_error_numeric, method='lm')

    # Generate x-values for the plot
    x = np.linspace(min(num_towers), max(num_towers), 300)

    # Compute the fitted curve using the optimized parameters
    fit_curve_tri = linear_model(x, params_tri[0], params_tri[1])

    # Plot the original data and the fitted curve
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.scatter(num_towers, absolute_mean_array_error_numeric, label='error_tri', s=30, c='b', marker='o')

    plt.xlabel('Number of Towers')
    plt.ylabel('Error')

    ax.set_yscale('asinh')
    ylim = 10**(-(precision-3))
    ax.set_ylim(bottom=-ylim*2, top=ylim*2)

    # Create a secondary y-axis
    ax2 = ax.twinx()

    # Plot the fitted curve with a different y-axis limit
    ax2.plot(x, fit_curve_tri, color='red', label='Fitted Curve (tri)')
    ax2.set_ylabel('Fitted Curve')

    # Set y-axis limits for the fitted curve plot
    ylim2 = 10**(-(precision-7))  # Set the desired y-axis limits for the fitted curve
    ax2.set_ylim(bottom=-ylim2, top=ylim2)

    # Combine the legends of both plots
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2)

    text_x = max(num_towers) * 0.5  # x-coordinate for the text annotation
    text_y = ax.get_ylim()[1] * 1.03  # y-coordinate for the text annotation
    a_value = params_tri[0]
    text = f'a = {a_value}'
    ax.text(text_x, text_y, text, fontsize=12, ha='center')

    plt.show()