import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import griddata

'''
DLD_Utils provides a necessary tasks from creating pillar
to handling results coming from the generated data
'''


class DLD_Utils:
    def add_mask(self, data, grid, pillar, mask_with=0):

        xy_mask, _ = pillar.to_mask(grid)

        x, y = self.square2parall(
            xy_mask[:, 0], xy_mask[:, 1], 1/pillar.N, pillar.D, pillar.G_X, G_R=pillar.G_R)
        xy_mask = np.concatenate(([x], [y])).T

        empty = np.empty((xy_mask.shape[0], data.shape[1]-xy_mask.shape[1]))
        empty[:] = mask_with

        mask_data = np.concatenate((xy_mask, empty), axis=1)

        return np.concatenate((data, mask_data))

    def insert_mask(self, data, grid, pillar, mask_with=0):

        _, idx = pillar.to_mask(grid)

        if len(data.shape) >= 2:
            shape = data.shape
            data = data.flatten()
            data[idx] = mask_with
            return data.reshape(shape)
        else:
            data[idx] = mask_with
            return data

    def parall2square(self, x, y, pillar):

        slope = 1 / pillar.N
        # Domain shear transformation from parallelogram to rectangular
        x_mapped = x
        y_mapped = y - slope * x

        # Domain transformation from rectangular to unitariy square
        X_mapped_MAX = pillar.D + pillar.G_X
        Y_mapped_MAX = pillar.D + pillar.G_X * pillar.G_R

        x_mapped = x_mapped / X_mapped_MAX
        y_mapped = y_mapped / Y_mapped_MAX

        return x_mapped, y_mapped

    def square2parall(self, x, y, pillar):

        slope = 1 / pillar.N

        X_MAX = pillar.D + pillar.G_X
        Y_MAX = pillar.D + pillar.G_X * pillar.G_R

        # Scaling square to rectangle
        x_mapped = x * X_MAX
        y_mapped = y * Y_MAX

        # Mapping rectangle to parallelogram by shear transformation
        x_mapped = x_mapped
        y_mapped = y_mapped + slope * x_mapped

        return x_mapped, y_mapped

    def interp2grid(self, x_data, y_data, data, x_grid, y_grid, method='linear', recover=False):
        # Interpolation of mapped data to x & y grid
        mapped = np.array([x_data, y_data]).T
        data_interp = griddata(mapped, data, (x_grid, y_grid), method=method)

        if recover:
            nearest = griddata(
                mapped, data, (x_grid, y_grid), method='nearest')
            data_interp[np.isnan(data_interp)] = nearest[np.isnan(data_interp)]

        return data_interp

    def compare_plots(self, data1, data2, figsize=(6, 6)):

        x, y, u, = data1[0], data1[1], data1[2]
        x_new, y_new, u_new, = data2[0], data2[1], data2[2]

        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_xlabel('$x$')
        ax1.set_ylabel('$y$')
        ax1.set_title("Before")
        plt.scatter(x, y, s=0.1, c=u)
        plt.colorbar()

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_xlabel('$x$')
        ax2.set_ylabel('$y$')
        ax2.set_title("After")
        plt.scatter(x_new, y_new, s=0.1, c=u_new)
        plt.colorbar()

        plt.show()

    def gradient(self, field, dx, dy, recover=True, plot=False, figsize=(8, 4)):

        grad_x = np.gradient(field, dx, axis=1)
        grad_y = np.gradient(field, dy, axis=0)

        if recover:
            grad_x = self.recover_gradient(grad_x)
            grad_y = self.recover_gradient(grad_y)

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            fig.subplots_adjust(left=0.1, wspace=0.5)

            im = axes[0].imshow(np.flip(grad_x, axis=0), extent=[
                                0, 1, 0, 1], cmap='rainbow')
            axes[0].set_title("grad_x")
            axes[0].set(xlabel="$x*$", ylabel="$y*$")

            xc = axes.flat[0].get_position().x0
            wc = axes.flat[0].get_position().width
            yc = axes.flat[0].get_position().y0
            hc = axes.flat[0].get_position().height
            cbar_ax = fig.add_axes([xc+wc+0.02, yc, 0.02, hc])
            fig.colorbar(im, cax=cbar_ax)

            im = axes[1].imshow(np.flip(grad_y, axis=0), extent=[
                                0, 1, 0, 1], cmap='rainbow')
            axes[1].set_title("grad_y")
            axes[1].set(xlabel="$x*$", ylabel="$y*$")

            cbar_ax = fig.add_axes([0.92, yc, 0.02, hc])
            fig.colorbar(im, cax=cbar_ax)

            plt.show()

        return grad_x, grad_y

    def recover_gradient(self, u, recover_with=0):

        sub_u_h = u[:, -3:]
        sub_u_f_h = np.flip(u, axis=1)[:, -3:]
        sub_u_v = u[-3:, :]
        sub_u_f_v = np.flip(u, axis=0)[-3:, :]

        sub_u_h[np.isnan(sub_u_h)] = sub_u_f_h[np.isnan(sub_u_h)]
        sub_u_v[np.isnan(sub_u_v)] = sub_u_f_v[np.isnan(sub_u_v)]

        u[:, -3:] = sub_u_h
        u[-3:, :] = sub_u_v

        u[np.isnan(u)] = recover_with

        return u

    def box_delete(self, array, MIN, MAX):
        
        min_array = np.min(array, axis=1)
        array_minimized = array[MIN <= min_array]
        max_array_minimized = np.max(array_minimized, axis=1)

        return array_minimized[max_array_minimized <= MAX]

    def wall_distance(self, grid, pillars, plot=True):

        X = np.array([])
        Y = np.array([])
        for pillar in pillars: 
            x, y = pillar.exterior.xy
            xp = np.asarray(x)
            yp = np.asarray(y)
            X = np.append(X, xp)
            Y = np.append(Y, yp)
            
        pillars_coord = np.array((X, Y)).T
        pillars_coord = self.box_delete(pillars_coord, 0, 1)
        _, mask_idx = self.pillar_mask(grid, pillar)
        
        domain_idx = np.setdiff1d(np.arange(len(grid[0].flatten())), mask_idx)
        domain_x_grid = grid[0].flatten()[domain_idx]
        domain_y_grid = grid[1].flatten()[domain_idx]
        
        wall_distance = np.zeros_like(grid[0])
        size_x = grid[0].shape[0]
        size_y = grid[0].shape[1]
        for x, y in zip(domain_x_grid, domain_y_grid):
            r = int(y * size_y)
            if r == size_y:
                r -= 1
            
            c = int(x * size_x)
            if c == size_x:
                c -= 1
            
            wall_distance[r, c] = np.amin(np.sqrt((x-pillars_coord[:, 0])**2 + (y-pillars_coord[:, 1])**2))

        if plot:
            fig = plt.figure()
            ax = plt.gca()
            im = plt.imshow(np.flip(wall_distance, axis=0), extent=[0, 1, 0, 1])
            xc = ax.get_position().x0
            wc = ax.get_position().width
            yc = ax.get_position().y0
            hc = ax.get_position().height
            cbar_ax = fig.add_axes([xc+wc+0.02, yc, 0.02, hc])
            fig.colorbar(im, cax=cbar_ax)

            plt.show()

        return wall_distance

    def compile_data(self, grid_size):

        x_grid_size = grid_size[0]
        y_grid_size = grid_size[1]

        xx = np.linspace(0, 1, x_grid_size)
        yy = np.linspace(0, 1, y_grid_size)
        x_grid, y_grid = np.meshgrid(xx, yy)

        directory = os.getcwd() + "\\Data"

        folders = [name for name in os.listdir(
            directory) if os.path.isdir(os.path.join(directory, name))]

        dataset_u = []
        dataset_v = []
        labels = []
        pbar1 = tqdm(total=len(folders), position=0, leave=True)
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

                label = list(map(float, name.split('_')))
                d, n, g, re = label[0], label[1], label[2], label[3]

                labels.append(label)

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

        return (np.array(dataset_u), np.array(dataset_v), np.array(labels))

    def save_data(self, data, name='data'):
        with open(name+".pickle", "wb") as f:
            pickle.dump(data, f)

    def load_data(self, name='data'):
        with open(name+".pickle", "rb") as f:
            return pickle.load(f)
