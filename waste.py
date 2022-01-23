
from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np
from shapely import affinity
from math import atan


D = 20
G_X = 40
G_R = 1
N = 10
pillar1 = Point((0, 0)).buffer(10)
pillar2 = affinity.translate(pillar1, xoff=D+G_X, yoff=(D+G_X*G_R)/N)
pillar3 = affinity.translate(pillar1, yoff=(D+G_X*G_R))
pillar4 = affinity.translate(pillar2, yoff=(D+G_X*G_R))

pillar1s = affinity.skew(pillar1, ys=-atan(1/N),
                         origin=(0, 0), use_radians=True)
pillar2s = affinity.skew(pillar2, ys=-atan(1/N),
                         origin=(0, 0), use_radians=True)
pillar3s = affinity.skew(pillar3, ys=-atan(1/N),
                         origin=(0, 0), use_radians=True)
pillar4s = affinity.skew(pillar4, ys=-atan(1/N),
                         origin=(0, 0), use_radians=True)

pillar1ss = affinity.scale(pillar1s, xfact=1/(D+G_X),
                           yfact=1/(D+G_X*G_R), zfact=1.0, origin=(0, 0))
pillar2ss = affinity.scale(pillar2s, xfact=1/(D+G_X),
                           yfact=1/(D+G_X*G_R), zfact=1.0, origin=(0, 0))
pillar3ss = affinity.scale(pillar3s, xfact=1/(D+G_X),
                           yfact=1/(D+G_X*G_R), zfact=1.0, origin=(0, 0))
pillar4ss = affinity.scale(pillar4s, xfact=1/(D+G_X),
                           yfact=1/(D+G_X*G_R), zfact=1.0, origin=(0, 0))

grid_size = (50, 50)
x_grid_size = grid_size[0]
y_grid_size = grid_size[1]

xx = np.linspace(0, 1, x_grid_size)
yy = np.linspace(0, 1, y_grid_size)
grid = np.meshgrid(xx, yy)

grid_points = np.array([grid[0].flatten(), grid[1].flatten()]).T
grid_Points = [Point(p) for p in grid_points.tolist()]


def contains(points):
    if pillar1ss.contains(points) or \
            pillar2ss.contains(points) or \
            pillar3ss.contains(points) or \
            pillar4ss.contains(points):
        return True
    else:
        return False


mask = filter(contains, grid_Points)
mask_xy = np.array([p.coords[0] for p in mask])

plt.scatter(mask_xy[:, 0], mask_xy[:, 1])
plt.show()
