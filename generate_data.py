import os
import mph
import csv
import time
from tqdm import tqdm
import numpy as np
from DLD_env import Pillar
from DLD_Utils import DLD_Utils as utl


def generate_data(simulator, f, N, Re):

    data_size = len(f)*len(N)*len(Re)

    # Import COMSOL model
    client = mph.start()
    pymodel = client.load(simulator)
    model = pymodel.java
    param = model.param()
    study = model.study('std1')
    result = model.result()
    cd = os.getcwd()

    folder = cd + "\\Data"
    os.makedirs(folder)
    info_f = list(map(str, f))
    info_N = list(map(str, N))
    info_Re = list(map(str, Re))

    info_f.insert(0, 'f')
    info_N.insert(0, 'N')
    info_Re.insert(0, 'Re')

    information = [info_f, info_N, info_Re]

    with open(folder + '\\information.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows(information)

    pbar = tqdm(total=data_size, position=0, leave=True)
    for ff in f:
        folder = cd + "\\Data\\f{}".format(ff)
        os.makedirs(folder)
        for n in N:
            for re in Re:
                # Set study's parameters
                param.set("f", str(ff))
                param.set("N", str(n))
                param.set("Re", str(re))

                # Run model
                study.run()

                # Export data
                filename = cd + \
                    "\\Data\\f{}\\{}_{}_{}.csv".format(
                        ff, ff, n, re)
                result.export("data1").set("filename", filename)
                result.export("data1").run()

                pbar.update(1)
                time.sleep(0.1)


def compile_data(grid_size):
    # Changing raw data into data that are compatable to our neural network
    x_grid_size = grid_size[0]
    y_grid_size = grid_size[1]

    xx = np.linspace(0, 1, x_grid_size)
    yy = np.linspace(0, 1, y_grid_size)
    x_grid, y_grid = np.meshgrid(xx, yy)

    dx = xx[1] - xx[0]
    dy = yy[1] - yy[0]

    directory = os.getcwd() + "\\Data"

    folders = [name for name in os.listdir(
        directory) if os.path.isdir(os.path.join(directory, name))]

    dataset_u = []
    dataset_v = []
    labels = []
    for folder, p1 in zip(folders, tqdm(range(len(folders)))):
        folder_dir = directory + "\\" + folder
        filesname = [os.path.splitext(filename)[0]
                     for filename in os.listdir(folder_dir)]

        time.sleep(0.1)

        for name, p2 in zip(filesname, tqdm(range(len(filesname)))):
            data = np.genfromtxt(
                folder_dir + "\\" + name + ".csv", delimiter=",")

            label = list(map(float, name.split('_')))
            ff, n, re = label[0], label[1], label[2]

            pillar = Pillar(ff, n)

            labels.append(label)

            x_mapped, y_mapped = utl.parall2square(
                data[:, 0], data[:, 1], pillar)

            u_interp = utl.interp2grid(x_mapped, y_mapped, data[:, 2],
                                       x_grid, y_grid, recover=True)

            v_interp = utl.interp2grid(x_mapped, y_mapped, data[:, 3],
                                       x_grid, y_grid, recover=True)

            # Make dataset
            dataset_u.append(u_interp)
            dataset_v.append(v_interp)

            time.sleep(0.1)

    return (np.array(dataset_u), np.array(dataset_v), np.array(labels))


if __name__ == "__main__":

    # f = np.round(np.linspace(0.25, 0.75, 26), 2).tolist()
    # N = [3, 4, 5, 6, 7, 8, 9, 10]
    # RE = [0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12.5, 15, 17.5, 20, 22.5, 25]
    # grid_size = (128, 128)

# data  test with integer N
    f = np.round(np.linspace(0.25, 0.75, 10), 2).tolist()
    N = [3, 4, 5, 6]
    RE = [0.05, 1.5, 6.5, 8.5, 12.5, 18.5]
    grid_size = (128, 128)
    # generate_data('DLD_COMSOL.mph', f, N, RE)

    dataset = compile_data(grid_size)
    utl.save_data(dataset, 'dataset_test_int')
