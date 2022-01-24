import os
import mph
import csv
import time
from tqdm import tqdm
from DLD_Utils import DLD_Utils as utl

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

D = [10, 15, 20, 25, 30, 35, 40, 45, 50]
N = [3, 4, 5, 6, 7, 8, 9, 10]
G = [10, 15, 20, 25, 30, 35, 40, 45, 50]
Re = [0.1, 0.2, 0.4, 0.8, 1, 2, 4, 6, 8, 10, 15, 20, 25, 30]

grid_size = (100, 100)

generate_data('DLD_COMSOL.mph', D, N, G, Re)
dataset = utl.compile_data(grid_size = grid_size)
utl.save_data(dataset, 'dataset')

