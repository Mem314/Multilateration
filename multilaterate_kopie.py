import numpy as np
import math

def get_locus(tower_1, tower_2, time_1, time_2, v, delta_d, max_d):
    x0 = []
    x1 = []
    x2 = []
    x3 = []

    t_delta_d = abs(time_1 - time_2)*v

    if(time_1 < time_2):
        sphere1 = (tower_1[0], tower_1[1], tower_1[2], 0)
        sphere2 = (tower_2[0], tower_2[1], tower_2[2], t_delta_d)
    else:
        sphere2 = (tower_2[0], tower_2[1], tower_2[2], 0)
        sphere1 = (tower_1[0], tower_1[1], tower_1[2], t_delta_d)
    for _ in range(int(max_d)//int(delta_d)):
        intersect = circle_intersection(circle1, circle2)
        if(intersect is not None):
            x0.append(intersect[0][0])
            x1.append(intersect[1][0])
            y0.append(intersect[0][1])
            y1.append(intersect[1][1])

        circle1 = (circle1[0], circle1[1], circle1[2] + delta_d)
        circle2 = (circle2[0], circle2[1], circle2[2] + delta_d)

    x0 = list(reversed(x0))
    y0 = list(reversed(y0))
    x = x0 + x1
    y = y0 + y1

    return [x, y]

def get_loci(rec_times, towers, v, delta_d, max_d):
    if (rec_times.shape[0] == 0):
        return []

    loci = []

    first_tower = int(np.argmin(rec_times))
    for j in [x for x in range(towers.shape[0]) if x!= first_tower]:
        print('tower', str(first_tower), 'to', str(j))
        locus = get_locus(tower_1=(towers[first_tower][0],towers[first_tower][1]),
                          tower_2=(towers[j][0],towers[j][1]),
                          time_1=rec_times[first_tower],
                          time_2=rec_times[j],
                          v = v, delta_d = delta_d, max_d = max_d)
        if(len(locus[0]) > 0):
            loci.append(locus)
    return loci

def circle_intersection(circle1, circle2):
    x1,y1,r1 = circle1
    x2,y2,r2 = circle2
    dx,dy = x2-x1,y2-y1
    d = math.sqrt(dx*dx+dy*dy)
    if d > r1+r2:
        print('no solution, the circles are seperate')
        return None
    elif d < abs(r1-r2):
        print('No solutions because one circle is contained within the other')
        return None
    elif d == 0 and r1 == r2:
        print('Circles are coincident - infinite number of solutions.')
        return None
    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h = math.sqrt(r1 * r1 - a * a)
    xm = x1 + a * dx / d
    ym = y1 + a * dy / d
    xs1 = xm + h * dy / d
    xs2 = xm - h * dy / d
    ys1 = ym - h * dx / d
    ys2 = ym + h * dx / d

    return ((xs1, ys1), (xs2, ys2))