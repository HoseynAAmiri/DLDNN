import os
import mph
import pickle
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from tqdm import tqdm
import Streamline_Particle as spt

'''
DLD_Util provides a variety of tasks from data generation
by given geometry and flow inputs based on the given
grid size
'''


class DLD_Util():
    def __init__(self):
        pass

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

        return np.nan_to_num(data_interp)

    def compare_plots(self, data1, data2, figsize=(6, 6)):

        x, y, u, v = data1[0], data1[1], data1[2], data1[3]
        x_new, y_new, u_new, v_new = data2[0], data2[1], data2[2], data2[3]

        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set_title("u (before)")
        plt.scatter(x, y, c=u)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_title("v (before)")
        plt.scatter(x, y, c=v)

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_title("u (after)")
        plt.scatter(x_new, y_new, c=u_new)
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_title("v (after)")
        plt.scatter(x_new, y_new, c=v_new)

        plt.show()

    def simulate_particle(self, u_interp, v_interp, start_point, no_period=1):

        shape = u_interp.shape
        xx = np.linspace(0, 1, shape[0])
        yy = np.linspace(0, 1, shape[1])
        x_grid, y_grid = np.meshgrid(xx, yy)

        stream = []
        # counter = np.zeros((no_period,))
        for i in range(no_period):
            stream.append(spt.streamplot(x_grid, y_grid, u_interp,
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

    def generate_data(self, simulator, D, N, G, Re):
        '''
        The setting in which our database is created
        D is the diameter of pillars
        G is the gap between pilarrs
        N is the periodic number of pillars lattice
        Re is Reynols number
        '''

        data_size = len(D)*len(N)*len(G)*len(Re)

        # Import COMSOL model
        client = mph.start()
        pymodel = client.load(simulator)
        self.model = pymodel.java
        self.param = self.model.param()
        self.study = self.model.study('std1')
        self.result = self.model.result()
        cd = os.getcwd()

        folder = cd + "\\Data"
        os.makedirs(folder)
        info_D = list(map(str, D))
        info_N = list(map(str, N))
        info_G = list(map(str, G))
        info_Re = list(map(str, Re))

        info_D.insert(0, 'D')
        info_N.insert(0, 'N')
        info_G.insert(0, 'G')
        info_Re.insert(0, 'Re')

        information = [info_D, info_N, info_G, info_Re]

        with open(folder + '\\information.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(information)

        pbar = tqdm(total=data_size, position=0, leave=True)

        for d in D:
            folder = cd + "\\Data\\D{}".format(d)
            os.makedirs(folder)
            for n in N:
                for g in G:
                    for re in Re:
                        # Set study's parameters
                        self.param.set("Do", str(d) + "[um]")
                        self.param.set("N", str(n))
                        self.param.set("G", str(g) + "[um]")
                        self.param.set("Re", str(re))

                        # Run model
                        self.study.run()

                        # Export data
                        filename = cd + \
                            "\\Data\\D{}\\{}_{}_{}_{}.csv".format(
                                d, d, n, g, re)
                        self.result.export("data1").set("filename", filename)
                        self.result.export("data1").run()

                        pbar.update(1)
                        time.sleep(0.1)

    def compile_data(self, grid_size=(100, 100)):

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
