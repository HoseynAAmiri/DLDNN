import numpy as np
import matplotlib.pyplot as plt
from DLD_Utils import DLD_Utils as utl


data = np.genfromtxt("psi_p.csv", delimiter=",")
N = 10
D = 20
G_X = 40
G_R = 1

grid_size = (100, 100)
x_grid_size = grid_size[0]
y_grid_size = grid_size[1]

xx = np.linspace(0, 1, x_grid_size)
yy = np.linspace(0, 1, y_grid_size)
x_grid, y_grid = np.meshgrid(xx, yy)

utl = utl(resolution=grid_size)
x_mapped, y_mapped = utl.parall2square(data[:, 0], data[:, 1], 1/N, D, G_X)
psi_mapped, p_mapped = utl.parall2square(data[:, 2], data[:, 3], 1/N, D, G_X)

data1 = tuple([x_mapped, y_mapped, psi_mapped, p_mapped])

xy_mask, idx = utl.pillar_mask((x_grid, y_grid), D, N, G_X)
# data_masked = utl.add_mask(data, xy_mask, mask_with=0)

x_mapped = np.concatenate((x_mapped, data[:, 0]))
y_mapped = np.concatenate((y_mapped, data[:, 1]))
psi_mapped = np.concatenate((psi_mapped, data[:, 2]))
p_mapped = np.concatenate((p_mapped, data[:, 3]))

psi_interp = utl.interp2grid(x_mapped, y_mapped, psi_mapped, x_grid, y_grid, method='nearest')
p_interp = utl.interp2grid(x_mapped, y_mapped, p_mapped, x_grid, y_grid, method='nearest')
psi_interp = utl.insert_mask(psi_interp, idx, mask_with=0)
p_interp = utl.insert_mask(p_interp, idx, mask_with=0)

data2 = tuple([x_grid.flatten(), y_grid.flatten(),
               psi_interp.flatten(), p_interp.flatten()])

compare = True
if compare:
    utl.compare_plots(data1, data2)
'''
x0 = 0
y0 = 0.5
point0 = np.array([x0, y0])
no_period = 50
stream = utl.simulate_particle(
    psi_interp, p_interp, start_point=point0, no_period=no_period)

# stream_original = stream
# for i in range(no_period):
#     stream_original[i][:, 0], stream_original[i][:, 1] = utl.square2parall(
#         stream[i][:, 0], stream[i][:, 1], 1/N, D, G_X)

utl.periodic_plot(stream)

'''