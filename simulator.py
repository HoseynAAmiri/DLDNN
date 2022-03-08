import numpy as np
from DLD_env import DLD_env, Pillar
from DLD_Utils import DLD_Utils as utl

# dataset = utl.load_data('dataset36')
# u_interp = dataset[0][0]
# v_interp = dataset[1][0]
# f, N, Re = dataset[2][0]

data = np.genfromtxt(".\\Data\\D0.2\\0.2_3_0.01.csv", delimiter=",")
f = 0.2
N = 3
Re = 0.01
grid_size = (128, 128)

pillar = Pillar(f, N)
dld = DLD_env(pillar, Re, resolution=grid_size)

x_mapped, y_mapped = utl.parall2square(data[:, 0], data[:, 1], pillar)

u, v = data[:, 2], data[:, 3]
data1 = tuple([x_mapped, y_mapped, u])

u_interp = utl.interp2grid(
    x_mapped, y_mapped, u, dld.x_grid, dld.y_grid, method='linear', recover=True)
v_interp = utl.interp2grid(
    x_mapped, y_mapped, v, dld.x_grid, dld.y_grid, method='linear', recover=True)

data2 = tuple([dld.x_grid.flatten(), dld.y_grid.flatten(), u_interp.flatten()])

compare = False
if compare:
    utl.compare_plots(data1, data2)

# import matplotlib.pyplot as plt
# plt.imshow(np.flip(psi_interp, axis=0), cmap='jet')
# ax = plt.gca()
# ax.get_xaxis().set_ticks([])
# ax.get_yaxis().set_ticks([])
# plt.show()


x0 = 0
y0 = 0.5
point0 = np.array([x0, y0])
periods = 7
d_particle = 0.1
stream = dld.simulate_particle(
    d_particle, (u_interp, v_interp), point0, periods=periods, plot=True)
