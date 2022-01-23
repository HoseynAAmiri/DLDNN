import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from scipy.interpolate import griddata
from shapely import affinity
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from math import atan
import particle_trajectory as ptj

'''
DLD_Utils provides a necessary tasks from creating pillar
to handling results from the generation codes
'''


class DLD_Utils:
    def __init__(self, resolution=(100, 100), pillar_type='circle'):
        self.pillar_type = pillar_type
        self.resolution = resolution

    def pillar_mask(self, grid, D, N, G_X, G_R=1):
        pillar1 = self.pillar(D)
        pillar2 = affinity.translate(pillar1, xoff=D+G_X, yoff=(D+G_X*G_R)/N)
        pillar3 = affinity.translate(pillar1, yoff=(D+G_X*G_R))
        pillar4 = affinity.translate(pillar2, yoff=(D+G_X*G_R))

        pillar1s = affinity.skew(
            pillar1, ys=-atan(1/N), origin=(0, 0), use_radians=True)
        pillar2s = affinity.skew(
            pillar2, ys=-atan(1/N), origin=(0, 0), use_radians=True)
        pillar3s = affinity.skew(
            pillar3, ys=-atan(1/N), origin=(0, 0), use_radians=True)
        pillar4s = affinity.skew(
            pillar4, ys=-atan(1/N), origin=(0, 0), use_radians=True)

        pillar1ss = affinity.scale(
            pillar1s, xfact=1/(D+G_X), yfact=1/(D+G_X*G_R), zfact=1.0, origin=(0, 0))
        pillar2ss = affinity.scale(
            pillar2s, xfact=1/(D+G_X), yfact=1/(D+G_X*G_R), zfact=1.0, origin=(0, 0))
        pillar3ss = affinity.scale(
            pillar3s, xfact=1/(D+G_X), yfact=1/(D+G_X*G_R), zfact=1.0, origin=(0, 0))
        pillar4ss = affinity.scale(
            pillar4s, xfact=1/(D+G_X), yfact=1/(D+G_X*G_R), zfact=1.0, origin=(0, 0))

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

        return mask_xy

    def pillar(self, D, pillar_org=(0, 0)):
        # First makes one pillar
        geometry_types = {'circle': 0, 'polygon': 1}
        if geometry_types.get(self.pillar_type) == 0:
            pillar = Point(pillar_org).buffer(D)
        else:
            pillar = Polygon([d for d in D])

        return pillar

    def parall2square(self, x, y, slope, D, G_X, G_R=1):
        # Domain shear transformation from parallelogram to a rectangular
        x_mapped = x
        y_mapped = y - slope * x

        # Domain transformation from rectangular to unitariy square
        X_mapped_MAX = D + G_X
        Y_mapped_MAX = D + G_X * G_R

        x_mapped = x_mapped / X_mapped_MAX
        y_mapped = y_mapped / Y_mapped_MAX

        return x_mapped, y_mapped

    def square2parall(self, x, y, slope, D, G_X, G_R=1):
        X_MAX = D + G_X
        Y_MAX = D + G_X * G_R

        # Scaling square to rectangle
        x_mapped = x * X_MAX
        y_mapped = y * Y_MAX

        # Mapping rectangle to parallelogram by shear transformation
        x_mapped = x_mapped
        y_mapped = y_mapped + slope * x_mapped

        return x_mapped, y_mapped

    def interp2grid(self, x_mapped, y_mapped, data_mapped, x_grid, y_grid, method='linear'):
        # Interpolation of mapped data to x & y grid
        mapped = np.array([x_mapped, y_mapped]).T
        data_interp = griddata(mapped, data_mapped,
                               (x_grid, y_grid), method=method)

        return data_interp

    def compare_plots(self, data1, data2, figsize=(6, 6)):

        x, y, u, v = data1[0], data1[1], data1[2], data1[3]
        x_new, y_new, u_new, v_new = data2[0], data2[1], data2[2], data2[3]

        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set_title("u (before)")
        plt.scatter(x, y, s=0.1, c=u)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_title("v (before)")
        plt.scatter(x, y, s=0.1, c=v)

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_title("u (after)")
        plt.scatter(x_new, y_new, s=0.1, c=u_new)
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_title("v (after)")
        plt.scatter(x_new, y_new, s=0.1, c=v_new)

        plt.show()

    def simulate_particle(self, u_interp, v_interp, start_point, no_period=1):

        shape = u_interp.shape
        xx = np.linspace(0, 1, shape[0])
        yy = np.linspace(0, 1, shape[1])
        x_grid, y_grid = np.meshgrid(xx, yy)

        stream = []
        # counter = np.zeros((no_period,))
        for i in range(no_period):
            stream.append(ptj.streamplot(x_grid, y_grid, u_interp,
                                         v_interp, start_point=start_point))

            if stream[i][-1, 0] >= 0.99:
                start_point = stream[i][-1, :] - [1, 0]
            elif stream[i][-1, 1] <= 0.01:
                start_point = stream[i][-1, :] + [0, 1]

        return stream

    def periodic_plot(self, stream, figsize=(6, 6)):

        fig = plt.figure(figsize=figsize)
        fig.add_subplot(1, 2, 1)
        no_period = len(stream)
        step_data = np.zeros((no_period, 2))
        for i in range(no_period):
            plt.plot(stream[i][:, 0], stream[i][:, 1], color=(
                0.1, 0.2, 0.5, (i/no_period+0.1)/1.1))

            step_data[i] = [stream[i][0, 1], stream[i][-1, 1]]

        fig.add_subplot(1, 2, 2)
        plt.plot(step_data[:, 0], step_data[:, 1])

        plt.show()

    def compile_data(self, grid_size=None):

        if not grid_size:
            grid_size = self.resolution

        x_grid_size = grid_size[0]
        y_grid_size = grid_size[1]

        xx = np.linspace(0, 1, x_grid_size)
        yy = np.linspace(0, 1, y_grid_size)
        x_grid, y_grid = np.meshgrid(xx, yy)

        directory = os.getcwd() + "\\Data"

        folders = [name for name in os.listdir(
            directory) if os.path.isdir(os.path.join(directory, name))]

        pbar1 = tqdm(total=len(folders), position=0, leave=True)

        dataset_u = []
        dataset_v = []
        labels = []
        for folder in folders:
            folder_dir = directory + "\\" + folder
            filesname = [os.path.splitext(filename)[0]
                         for filename in os.listdir(folder_dir)]

            pbar1.update(1)
            time.sleep(0.1)

            pbar2 = tqdm(total=len(filesname), position=0, leave=True)

            for name in filesname:
                data = np.genfromtxt(
                    folder_dir + "\\" + name + ".csv", delimiter=",")
                data = np.nan_to_num(data)

                name = list(map(float, name.split('_')))
                d, n, g, re = name[0], name[1], name[2], name[3]

                labels.append(name)

                x_mapped, y_mapped = self.parall2square(
                    data[:, 0], data[:, 1], 1/n, d, g)
                u_mapped, v_mapped = self.parall2square(
                    data[:, 2], data[:, 3], 1/n, d, g)

                u_interp = self.interp2grid(x_mapped, y_mapped, u_mapped,
                                            x_grid, y_grid)
                v_interp = self.interp2grid(x_mapped, y_mapped, v_mapped,
                                            x_grid, y_grid)

                # Make dataset
                dataset_u.append(u_interp)
                dataset_v.append(v_interp)

                pbar2.update(1)
                time.sleep(0.1)

        xyuv = (np.array(dataset_u), np.array(dataset_v), np.array(labels))
        return xyuv

    def save_data(self, data, name='data'):
        with open(name+".pickle", "wb") as f:
            pickle.dump(data, f)

    def load_data(self, name='data'):
        with open(name+".pickle", "rb") as f:
            return pickle.load(f)
