import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import sympy as sy


abstand = [[1.5, 1.8, 2.0, 3.08], [-3, -2.2, 3.5, 5.1], [-2.3, 1.4, -2.7, 3.81], [3.3, -2.5, -3.2, 5.2], [1.5,1.1,1.2]]
#abstand = [[-0.5,0.5,0,1],[-1,0.5,0,1],[-1,-0.5,0,1],[-1.5,0.2,0,1]]
#abstand = [[0,0,0,1],[2,0,0,1],[1,1,0,1],[1,-1,0,1]]
#abstand = [[0,0,0],[2,0,0],[1,1,0],[1,-1,0],[0,0,0],[-2,0,0],[-1,1,0],[-1,-1,0]]
distances = [3.08,5.1,3.81,5.2,1]


def Trilateration_3D(abstand, distances):
    """
    Dieses Programm implementiert die Trilateration in 3 Dimensionen.
    Ziel ist es, die Position eines Punktes (posi) im Raum auf der Grundlage der
    Position (Koordinaten) von P_i anderen Punkten im Raum und der Entfernung
    des unbekannten Punktes von diesen P_i bekannten Punkten zu bestimmen.

    Man benötigt mindestens 4-Kugeln, um posi zu bestimmen. Denn der
    schnittpunkt von den ersten zwei Kugeln ist ein Kreis. Der Schnittpunkt von
    zwei Kreisen ist eine Geodäte. Damit werden mindestens 4-Kugeln benötigen,
    durch deren Schnittpunkte erhält man 3 Kreise und durch deren Schnittpunkte
    erhält man 2 Geodäten. Durch den Schnittpunkt der Geodäten, ist die Position
    von posi ermittelt.
    Für 4-Kugeln erhält man eine Geodäte ==> für (4+n)-Kugeln erhält man (n+1)-Geodäten.

    Hier handelt es sich um eine kartesische 'true-range-Multilateration'.
    Durch den ersten Punkt und mithilfe des Pythagoras-Theorems ist man in der Lage,
    die Abstände zwischen posi und Mittelpunkt der Kugeln zu berechnen.
    """

    # coordinates: p1 = [x, y, z] and radii.
    p = []
    for i in range(len(abstand)):
        p.append(np.array(abstand[i][:3]))
    r = np.array(distances)
    # the unit vector in the direction from p1 to p2.
    d, i, j, e_x, e_y, e_z = [], [], [], [], [], []
    for k in range(len(abstand)-3):
        d.append(np.linalg.norm(p[k + 1] - p[k]))
        # projection of the vector from p3 to p1 onto e_x.
        e_x.append((p[k + 1] - p[k]) / np.linalg.norm(p[k + 1] - p[k]))
        i.append(np.dot(e_x[k], (p[k + 2] - p[k])))
        e_y.append((p[k + 2] - p[k] - (i[k] * e_x[k])) / (np.linalg.norm(p[k + 2] - p[k] - (i[k] * e_x[k]))))
        j.append(np.dot(e_y[k], (p[k + 2] - p[k])))
        e_z.append(np.cross(e_x[k], e_y[k]))

    x, y, z1, z2 = [], [], [], []
    for k in range(len(abstand) - 3):
        x_val = ((r[k] ** 2) - (r[k + 1] ** 2) + (d[k] ** 2)) / (2 * d[k])
        x.append(x_val)
        y_val = (((r[k] ** 2) - (r[k + 2] ** 2) + (i[k] ** 2) + (j[k] ** 2)) / (2 * j[k])) - ((i[k] / j[k]) * (x_val))
        y.append(y_val)
        z1.append(np.sqrt(np.abs(r[k] ** 2 - x_val ** 2 - y_val ** 2)))
        z2.append(z1[k] * (-1))

    ans1, ans2, dist1, dist2 = [], [], [], []
    for k in range(len(abstand) - 3):
        ans1.append(p[k] + (x[k] * e_x[k]) + (y[k] * e_y[k]) + (z1[k] * e_z[k]))
        ans2.append(p[k] + (x[k] * e_x[k]) + (y[k] * e_y[k]) + (z2[k] * e_z[k]))
        dist1.append(np.linalg.norm(p[k + 3] - ans1[k]))
        dist2.append(np.linalg.norm(p[k + 3] - ans2[k]))

    positions = []
    for k in range(len(abstand) - 3):
        if np.abs(r[k + 3] - dist1[k]) < np.abs(r[k + 3] - dist2[k]):
            positions.append(ans1[k])
        else:
            positions.append(ans2[k])

    return positions

if __name__ == "__main__":
    # Print out the data
    print("The input four points and distances, in the format of [x, y, z, d], are:")
    for p in range(0, len(abstand)):
        print(abstand[p])

    decimal_places = 12
    positions = Trilateration_3D(abstand, distances)
    def format_positions(posi, decimal_places):
        formatted_values = [("[{}]".format(", ".join(["{:.{}f}".format(x, decimal_places) for x in pos.tolist()]))) for
                            pos in posi]
        return formatted_values

    formatted_positions = format_positions(positions, decimal_places=12)
    for pos in formatted_positions:
        print("Position: {}".format(pos))

    mean_position = np.mean(positions, axis=0)
    print("mean of the positions: {}".format(mean_position))