import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

'''
DLD_Utils provides a necessary tasks from creating pillar
to handling results coming from the generated data
'''


class DLD_Utils:    
    @classmethod
    def add_mask(self, data, grid, pillar, mask_with=0):

        xy_mask, _ = pillar.to_mask(grid)

        x, y = self.square2parall(
            xy_mask[:, 0], xy_mask[:, 1], pillar)
        xy_mask = np.concatenate(([x], [y])).T

        empty = np.empty((xy_mask.shape[0], data.shape[1]-xy_mask.shape[1]))
        empty[:] = mask_with

        mask_data = np.concatenate((xy_mask, empty), axis=1)

        return np.concatenate((data, mask_data))

    @staticmethod
    def insert_mask(data, grid, pillar, mask_with=0):

        _, idx = pillar.to_mask(grid)

        if len(data.shape) >= 2:
            shape = data.shape
            data = data.flatten()
            data[idx] = mask_with
            return data.reshape(shape)
        else:
            data[idx] = mask_with
            return data
    
    @staticmethod
    def parall2square(x, y, pillar):
    # This function transform data from parallolegram into unitary square domain
        slope = 1 / pillar.N
        # Domain shear transformation from parallelogram to rectangular
        x_mapped = x
        y_mapped = y - slope * x

        # Domain transformation from rectangular to unitariy square
        Y_mapped_MAX = pillar.SIZE + (1 - pillar.SIZE) * pillar.G_R

        y_mapped = y_mapped / Y_mapped_MAX

        return x_mapped, y_mapped

    @staticmethod
    def square2parall(x, y, pillar):
        # This function transform data from unitary square into the original parallelogeram domain 
        slope = 1 / pillar.N

        Y_MAX = pillar.SIZE + (1 - pillar.SIZE) * pillar.G_R

        # Scaling square to rectangle
        x_mapped = x
        y_mapped = y * Y_MAX

        # Mapping rectangle to parallelogram by shear transformation
        y_mapped = y_mapped + slope * x_mapped

        return x_mapped, y_mapped

    @staticmethod
    def interp2grid(x_data, y_data, data, x_grid, y_grid, method='linear', recover=False):
    # This function interpolate the data from any grid to the unitary grid     
        # Interpolation of mapped data to x & y grid
        mapped = np.array([x_data, y_data]).T
        data_interp = griddata(mapped, data, (x_grid, y_grid), method=method)

        if recover:
            nearest = griddata(
                mapped, data, (x_grid, y_grid), method='nearest')
            data_interp[np.isnan(data_interp)] = nearest[np.isnan(data_interp)]

        return data_interp
    
    @staticmethod
    def compare_plots(data1, data2, figsize=(6, 3)):
    # Any two sets of data can be compared by this function
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

    @classmethod
    def gradient(self, field, dx, dy, recover=False):
    # Gradiant function simply gets a field as input and compute it's gradiant in x and y direction
    # here we used this function in two intences: first, coputing velocity vectors from psi field and
    # determining the normal vectors of pillars from wall distance function
        grad_x = np.gradient(field, dx, axis=1)
        grad_y = np.gradient(field, dy, axis=0)

        if recover:
            grad_x = self.recover_gradient(grad_x)
            grad_y = self.recover_gradient(grad_y)

        
        return grad_x, grad_y

    @staticmethod
    def recover_gradient(u, recover_with=0):
    # this function fill the places in domain where gradiant was not computable 
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

    @staticmethod
    def box_delete(array, MIN, MAX):
    # box delete function trim pillars coordinate out of the domain    
        min_array = np.min(array, axis=1)
        array_minimized = array[MIN <= min_array]
        max_array_minimized = np.max(array_minimized, axis=1)

        return array_minimized[max_array_minimized <= MAX]

    @staticmethod
    def load_data(name='data'):
        with open(name+".pickle", "rb") as file:
                return pickle.load(file)

    @staticmethod    
    def save_data(data, name='data'):
        with open(name+".pickle", "wb+") as file:
            pickle.dump(data, file)

