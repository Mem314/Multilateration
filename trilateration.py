import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import sympy as sy


abstand = [[1.5, 1.8, 2.0, 3.08], [-3, -2.2, 3.5, 5.1], [-2.3, 1.4, -2.7, 3.81], [3.3, -2.5, -3.2, 5.2]]
#abstand = [[-0.5,0.5,0,1],[-1,0.5,0,1],[-1,-0.5,0,1],[-1.5,0.2,0,1]]
#abstand = [[0,0,0,1],[2,0,0,1],[1,1,0,1],[1,-1,0,1]]
#abstand = [[-1,0,0,1],[1,0,0,1],[0,1,0,1],[0,-1,0,1]]
distances = [3.08, 5.1, 3.81, 5.2]

x0, y0, z0 = [], [], []
for i in range(0, len(abstand)):
    x0.__iadd__([abstand[i][0]])
    y0.__iadd__([abstand[i][1]])
    z0.__iadd__([abstand[i][2]])

dx, dy, dz = [], [], []
for i in range(1, len(abstand)):
    dx.__iadd__([abstand[i][0] - abstand[0][0]])
    dy.__iadd__([abstand[i][1] - abstand[0][1]])
    dz.__iadd__([abstand[i][2] - abstand[0][2]])

dr = []
for i in range(0, len(abstand)-1):
    dr.__iadd__([np.sqrt(dx[i]**2 + dy[i]**2 + dz[i]**2)])

r = []
for i in range(0, len(abstand)):
    r.__iadd__([abstand[i][3]])


def Trilateration_3D(abstand, distances):
    '''
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
    '''

    # Koordinaten: p1 = [x, y, z].
    p1 = np.array(abstand[0][:3])
    p2 = np.array(abstand[1][:3])
    p3 = np.array(abstand[2][:3])
    p4 = np.array(abstand[3][:3])

    # Radien: r1 ist Radius von p1.
    r1 = distances[0]
    r2 = distances[1]
    r3 = distances[2]
    r4 = distances[3]

    # Einheitsvektor des ersten Punktes.
    e_x = (p2 - p1) / np.linalg.norm(p2 - p1)

    # x koordinate des dritten Punktes.
    i = np.dot(e_x, (p3 - p1))

    # Einheitsvektor des zweiten und dritten Punkt.
    e_y = (p3 - p1 - (i * e_x)) / (np.linalg.norm(p3 - p1 - (i * e_x)))
    e_z = np.cross(e_x, e_y)

    # Abstand von den ersten zwei Punkten.
    d = np.linalg.norm(p2 - p1)

    # y koordinate des dritten Punktes.
    j = np.dot(e_y, (p3 - p1))

    # Die Koordinaten des zu bestimmenden Punktes, wobei für z zwei Lsgen gibt.
    x = ((r1 ** 2) - (r2 ** 2) + (d ** 2)) / (2 * d)
    y = (((r1 ** 2) - (r3 ** 2) + (i ** 2) + (j ** 2)) / (2 * j)) - ((i / j) * (x))
    z1 = np.sqrt(np.abs(r1 ** 2 - x ** 2 - y ** 2))
    z2 = z1 * (-1)

    # Ansätze für die Lsgen. Diese sind abhängig von der Anzahl der z-Lsgen.
    ans1 = p1 + (x * e_x) + (y * e_y) + (z1 * e_z)
    ans2 = p1 + (x * e_x) + (y * e_y) + (z2 * e_z)

    # Die Lösungen verbinden mit dem 4.ten Punkt.
    dist1 = np.linalg.norm(p4 - ans1)
    dist2 = np.linalg.norm(p4 - ans2)

    # Ausgabe der richtigen Lsg.
    if np.abs(r4 - dist1) < np.abs(r4 - dist2):
        return ans1
    else:
        return ans2
if __name__ == "__main__":
    # Print out the data
    print("The input four points and distances, in the format of [x, y, z, d], are:")
    for p in range(0, len(abstand)):
        print(abstand[p])


    # Call the function and compute the location
    posi = Trilateration_3D(abstand, distances)
    decimal_places = 10
    formatted_values = [("{:.{}f}".format(x, decimal_places)) for x in posi]
    formatted_string = ", ".join(formatted_values)
    print("The locations of the points are:", formatted_string)