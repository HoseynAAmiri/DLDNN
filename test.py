import numpy as np
import matplotlib.pyplot as plt
from DLD_Utils import DLD_Utils as utl

utl = utl()
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

x_mapped, y_mapped = utl.parall2square(data[:, 0], data[:, 1], 1/N, D, G_X)
u_mapped, v_mapped = utl.parall2square(data[:, 2], data[:, 3], 1/N, D, G_X)

data1 = tuple([x_mapped, y_mapped, u_mapped, v_mapped])

xy_mask, idx = utl.pillar_mask((x_grid, y_grid), D, N, G_X)
# data_masked = utl.add_mask(data, xy_mask, mask_with=0)

x_mapped = np.concatenate((x_mapped, data[:, 0]))
y_mapped = np.concatenate((y_mapped, data[:, 1]))
u_mapped = np.concatenate((u_mapped, data[:, 2]))
v_mapped = np.concatenate((v_mapped, data[:, 3]))

u_interp = utl.interp2grid(x_mapped, y_mapped, u_mapped, x_grid, y_grid, method='nearest')
v_interp = utl.interp2grid(x_mapped, y_mapped, v_mapped, x_grid, y_grid, method='nearest')
u_interp = utl.insert_mask(u_interp, idx, mask_with=0)
v_interp = utl.insert_mask(v_interp, idx, mask_with=0)

data2 = tuple([x_grid.flatten(), y_grid.flatten(),
               u_interp.flatten(), v_interp.flatten()])

compare = True
if compare:
    utl.compare_plots(data1, data2)

'''
x_original, y_original = utl.square2parall(x_grid, y_grid, 1/N, D, G_X)
u_original, v_original = utl.square2parall(u_interp, v_interp, 1/N, D, G_X)

data3 = tuple([x_original, y_original, u_original, v_original])

if compare:
    utl.compare_plots(data2, data3)

x0 = 0
y0 = 0.5
point0 = np.array([x0, y0])
no_period = 50
stream = utl.simulate_particle(
    u_interp, v_interp, start_point=point0, no_period=no_period)

# stream_original = stream
# for i in range(no_period):
#     stream_original[i][:, 0], stream_original[i][:, 1] = utl.square2parall(
#         stream[i][:, 0], stream[i][:, 1], 1/N, D, G_X)

utl.periodic_plot(stream)
'''
