import os
import mph
import csv
import time
from tqdm import tqdm
import pickle
import numpy as np
from DLD_env import DLD_env, Pillar
from DLD_Utils import DLD_Utils as utl
utl = utl()
'''
The settings that creates database
D is the diameter of pillars
G is the gap between pillars
N is the periodic number of pillars lattice
Re is the Reynols number
'''


def generate_data(simulator, D, N, G, Re):

    data_size = len(D)*len(N)*len(G)*len(Re)

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
                    param.set("D", str(d) + "[um]")
                    param.set("N", str(n))
                    param.set("G", str(g) + "[um]")
                    param.set("Re", str(re))

                    # Run model
                    study.run()

                    # Export data
                    filename = cd + \
                        "\\Data\\D{}\\{}_{}_{}_{}.csv".format(
                            d, d, n, g, re)
                    result.export("data1").set("filename", filename)
                    result.export("data1").run()

                    pbar.update(1)
                    time.sleep(0.1)

# Changing raw data into data that are compatable to our neural network


def compile_data(grid_size):

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

    dataset_psi = []
    dataset_px = []
    dataset_py = []
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
            d, n, g, re = label[0], label[1], label[2], label[3]

            pillar = Pillar(d, n, g)

            labels.append(label)

            x_mapped, y_mapped = utl.parall2square(
                data[:, 0], data[:, 1], pillar)

            psi_interp = utl.interp2grid(x_mapped, y_mapped, data[:, 2],
                                         x_grid, y_grid, recover=True)

            p_interp = utl.interp2grid(x_mapped, y_mapped, data[:, 3],
                                       x_grid, y_grid, recover=True)

            px, py = utl.gradient(p_interp, dx, dy)
            px = utl.insert_mask(px, (x_grid, y_grid), pillar)
            py = utl.insert_mask(py, (x_grid, y_grid), pillar)

            # Make dataset
            dataset_psi.append(psi_interp)
            dataset_px.append(px)
            dataset_py.append(py)

            time.sleep(0.1)

    return (np.array(dataset_psi), np.array(dataset_px), np.array(dataset_py), np.array(labels))

if __name__ == "__main__":
    '''
    D = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    N = [3, 4, 5, 6, 7, 8, 9, 10]
    G = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    Re = [0.01, 0.1, 0.2, 0.4, 0.8, 1, 2, 4, 6, 8, 10, 15, 20, 25, 30]
    '''

    D = [10, 20, 30, 40, 50]
    N = [3, 5, 7, 9]
    G = [10, 20, 30, 40, 50]
    RE = [0.01, 0.1, 0.2, 0.4, 0.8, 1]

    grid_size = (128, 128)

    # generate_data('DLD_COMSOL.mph', D, N, G, RE)

    dataset = compile_data(grid_size)
    utl.save_data(dataset, 'dataset')
