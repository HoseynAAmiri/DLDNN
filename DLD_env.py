import numpy as np
import matplotlib.pyplot as plt
from math import atan
from shapely import affinity
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from descartes import PolygonPatch
import particle_trajectory as ptj
from DLD_Utils import DLD_Utils as utl

class DLD_env:
    def __init__(self, pillar, Re, resolution=(100, 100)):

        self.pillar = pillar
        self.Re = Re    
        self.resolution = resolution
        self.x_grid, self.y_grid, self.dx, self.dy = self.grid()

    def grid(self, grid_size=None):

        if not grid_size:
            grid_size = self.resolution

        x_grid_size = grid_size[0]
        y_grid_size = grid_size[1]

        xx = np.linspace(0, 1, x_grid_size)
        yy = np.linspace(0, 1, y_grid_size)
        x_grid, y_grid = np.meshgrid(xx, yy)

        dx = xx[1] - xx[0]
        dy = yy[1] - yy[0]

        return x_grid, y_grid, dx, dy
    
    def simulate_particle(self, dp, uv, pillars, start_point, periods=1, plot=False, figsize=(9, 4)):

        wall_distance = self.wallfunc((self.x_grid, self.y_grid), pillars, plot=False)
        ny, nx = self.gradient(wall_distance, self.dx, self.dy, recover=True, plot=False)

        dist_mag = np.ma.sqrt(nx**2 + ny**2)
        nx = - nx / dist_mag
        ny = ny / dist_mag

        stream = []
        for i in range(periods):
            stream.append(ptj.streamplot((self.x_grid, self.y_grid), uv, (nx, ny), pillars, dp, start_point))

            if stream[i][-1, 0] >= 0.99:
                start_point = stream[i][-1, :] - [1, 0]
            elif stream[i][-1, 1] <= 0.01:
                start_point = stream[i][-1, :] + [0, 1]

        if plot:
            fig = plt.figure(figsize=figsize)
            fig.add_subplot(1, 2, 1)
            periods = len(stream)
            step_data = np.zeros((periods, 2))
            for i in range(periods):
                plt.plot(stream[i][:, 0], stream[i][:, 1], color=(
                    0.1, 0.2, 0.5, (i/periods+0.1)/1.1))

                step_data[i] = [stream[i][0, 1], stream[i][-1, 1]]

            plt.xlim([0, 1])
            plt.ylim([0, 1])

            ax = plt.gca()
            for pillar in pillars:
                ax.add_patch(PolygonPatch(pillar, fc='red'))
                ax.add_patch(PolygonPatch(pillar.buffer(
                    dp/2).difference(pillar), fc='white', ec='#999999'))

            fig.add_subplot(1, 2, 2)
            plt.plot(step_data[:, 0], step_data[:, 1])
            plt.xlim([0, 1])
            plt.ylim([0, 1])

            plt.show()

        return stream

    
class Pillar:
    def __init__(self, size, N, G_X, G_R=1, pillar_type='circle', origin=(0,0)):

        self.size = size
        self.pillar_type = pillar_type
        
        geometry_types = {'circle': 0, 'square': 1}

        if geometry_types.get(pillar_type) == 0:
            self.pillar = Point(origin).buffer(self.size/2)

        elif geometry_types.get(pillar_type) == 1:
            self.pillar = Polygon([(self.size/2, self.size/2),
                                    (self.size/2, -self.size/2),
                                    (-self.size/2, -self.size/2),
                                    (-self.size/2, self.size/2)])
            
        else: raise ValueError("The input pillar type is invalid")
        
        self.pillar = affinity.translate(self.pillar, xoff=origin[0], yoff=origin[1])
        self.N, self.G_X, self.G_R = N, G_X, G_R
        self.pillars = self.to_pillars()

    def to_pillars(self):

        pillar1 = self.pillar
        pillar2 = affinity.translate(pillar1, xoff=pillar1.size+self.G_X, yoff=(pillar1.size+self.G_X*self.G_R)/self.N)
        pillar3 = affinity.translate(pillar1, yoff=(pillar1.size+self.G_X*self.G_R))
        pillar4 = affinity.translate(pillar2, yoff=(pillar1.size+self.G_X*self.G_R))

        pillar1s = affinity.skew(
            pillar1, ys=-atan(1/self.N), origin=(0, 0), use_radians=True)
        pillar2s = affinity.skew(
            pillar2, ys=-atan(1/self.N), origin=(0, 0), use_radians=True)
        pillar3s = affinity.skew(
            pillar3, ys=-atan(1/self.N), origin=(0, 0), use_radians=True)
        pillar4s = affinity.skew(
            pillar4, ys=-atan(1/self.N), origin=(0, 0), use_radians=True)

        pillar1ss = affinity.scale(
            pillar1s, xfact=1/(pillar1.size+self.G_X), yfact=1/(pillar1.size+self.G_X*self.G_R), zfact=1.0, origin=(0, 0))
        pillar2ss = affinity.scale(
            pillar2s, xfact=1/(pillar1.size+self.G_X), yfact=1/(pillar1.size+self.G_X*self.G_R), zfact=1.0, origin=(0, 0))
        pillar3ss = affinity.scale(
            pillar3s, xfact=1/(pillar1.size+self.G_X), yfact=1/(pillar1.size+self.G_X*self.G_R), zfact=1.0, origin=(0, 0))
        pillar4ss = affinity.scale(
            pillar4s, xfact=1/(pillar1.size+self.G_X), yfact=1/(pillar1.size+self.G_X*self.G_R), zfact=1.0, origin=(0, 0))

        pillars = [pillar1ss, pillar2ss, pillar3ss, pillar4ss]

        return pillars

    def to_mask(self, grid):

        pillars = self.pillars

        grid_points = np.array([grid[0].flatten(), grid[1].flatten()]).T
        grid_Points = [Point(p) for p in grid_points.tolist()]

        def contains(point):

            inside = False
            for pillar in pillars:
                if pillar.contains(point):
                    inside = True

            return inside

        mask = filter(contains, grid_Points)
        idx = list(filter(lambda i: contains(
            grid_Points[i]), range(len((grid_Points)))))
        xy_mask = [p.coords[0] for p in mask]

        return np.array(xy_mask), idx

