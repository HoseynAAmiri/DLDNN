from DLD_Util import DLD_Util


D = [10, 15, 20, 25, 30, 35, 40, 45, 50]
N = [3, 4, 5, 6, 7, 8, 9, 10]
G = [10, 15, 20, 25, 30, 35, 40, 45, 50]
Re = [0.1, 0.2, 0.4, 0.8, 1, 2, 4, 6, 8, 10, 15, 20, 25, 30]

grid_size = (100, 100)

m = DLD_Util()
# m.generate_data('DLD_COMSOL.mph', D, N, G, Re)
dataset = m.compile_data(grid_size = grid_size)
m.save_data(dataset, 'dataset')
# dataset = m.load_data('dataset')
