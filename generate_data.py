import os
import mph
import csv
import time
from tqdm import tqdm
import pickle
import numpy as np
from DLD_Utils import DLD_Utils as utl
utl = utl()
'''
The settings that creates database
D is the diameter of pillars
G is the gap between pilarrs
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
                    param.set("Do", str(d) + "[um]")
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

            x_mapped, y_mapped = utl.parall2square(
                data[:, 0], data[:, 1], 1/n, d, g)
            u_mapped, v_mapped = utl.parall2square(
                data[:, 2], data[:, 3], 1/n, d, g)

            u_interp = utl.interp2grid(x_mapped, y_mapped, u_mapped,
                                        x_grid, y_grid)
            v_interp = utl.interp2grid(x_mapped, y_mapped, v_mapped,
                                        x_grid, y_grid)

            # Make dataset
            dataset_u.append(u_interp)
            dataset_v.append(v_interp)

            pbar2.update(1)
            time.sleep(0.1)

    return (np.array(dataset_u), np.array(dataset_v), np.array(labels))

def save_data(data, name='data'):
    with open(name+".pickle", "wb") as f:
        pickle.dump(data, f)

D = [10, 15, 20, 25, 30, 35, 40, 45, 50]
N = [3, 4, 5, 6, 7, 8, 9, 10]
G = [10, 15, 20, 25, 30, 35, 40, 45, 50]
Re = [0.1, 0.2, 0.4, 0.8, 1, 2, 4, 6, 8, 10, 15, 20, 25, 30]

grid_size = (100, 100)

generate_data('DLD_COMSOL.mph', D, N, G, Re)
dataset = compile_data(grid_size = grid_size)
save_data(dataset, 'dataset')

