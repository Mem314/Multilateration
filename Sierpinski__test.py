import matplotlib.pyplot as plt
import numpy as np
import random
import math
from matplotlib.animation import FuncAnimation

The_Sierpinski_Quadrilateral = True
length = 30

fig = plt.figure()
ax = fig.add_subplot()
max_width = max(0, length)
ax.set_ylim((0, max_width))
ax.set_xlim((0, max_width))
#ax.plot((-max_width + 1, max_width - 1), (0, 0), (0, 0), 'r', label='x-axis')
#ax.plot((0, 0), (-max_width + 1, max_width - 1), (0, 0), 'k', label='y-axis')

class Points:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color

    def coord(self):
        return self.x, self.y
    def plot_points(self):
        plot = plt.plot(self.x, self.y, marker=".", color=self.color)
        return plot


p = length
point_1 = Points(1,1, "red")
point_2 = Points(p-1,1, "red")
point_3 = Points(1, p-1, "red")
point_4 = Points(p-1, p-1, "red")
point_1.plot_points()
point_2.plot_points()
point_3.plot_points()
point_4.plot_points()

# number of the points.
num_points = np.array(range(1, 100))

a0 = p-1  # upper bound
b0 = 1    # lower bound

#first_point = ((np.random.rand(num_points, 2)) * (b-a)) + a

def first_point(radius):
    """
    Random first point within a circle with the given radius.
    """
    angle = 2 * math.pi * random.random()  # Random angle in radians
    distance = radius * math.sqrt(random.random())  # Random distance from the center
    point_x = distance * math.cos(angle)
    point_y = distance * math.sin(angle)
    return (point_x, point_y)

circle_radius = 30  # Radius of the circle

point = [first_point(circle_radius) for _ in range(1)]
e1, e2 = zip(*point)
plt.scatter(e1, e2, s=1)


if The_Sierpinski_Quadrilateral:
    def distance(t1, t2):
        return math.sqrt((t1[0] - t2[0])**2 + (t1[1] - t2[1])**2)

    distance(point[0], random.choice((point_1.coord(), point_2.coord(), point_3.coord(), point_4.coord()))) / 2

    def middle_point(t1, t2):
        x = (t1[0] + t2[0])/2
        y = (t1[1] + t2[1])/2
        return list(np.append(x, y))


    num_points_ann = ax.text(15, 32, 'number of points = 0')

    # start point
    random_eck_point_0 = random.choice((point_1.coord(), point_2.coord(), point_3.coord(), point_4.coord()))
    x0 = middle_point(point[0], random_eck_point_0)[0]
    y0 = middle_point(point[0], random_eck_point_0)[1]
    xi, yi = [], []
    xi.append(x0)
    yi.append(y0)

    n_frames = num_points

    def animate(j):
        global n_frames,num_points_ann

        ax.collections.clear()

        # the other points
        random_eck_point_i = random.choice((point_1.coord(), point_2.coord(), point_3.coord(), point_4.coord()))
        x = middle_point((xi[j - 1], yi[j - 1]), random_eck_point_i)[0]
        xi .append(x)

        y = middle_point((xi[j - 1], yi[j - 1]), random_eck_point_i)[1]
        yi .append(y)
        plt.scatter(xi, yi, s=2, c='#1f77b4')

        num_points_ann.set_text('number of points = {:} '.format(j))
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=1, blit=False, repeat=True)
    #anim.save('C:/Users/Mem/Desktop/Studium/Vertiefungsmodul/Animationen/The_Sierpinski_Triangle.gif', writer='imagemagick', fps=60)
    #anim.save('/home/mohammed/Animationen/The_Sierpinski_Triangle.gif', writer='imagemagick', fps=60)

    plt.show()

