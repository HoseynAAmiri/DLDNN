import numpy as np
import matplotlib.pyplot as plt
from DLD_Utils import DLD_Utils as utl

data = np.genfromtxt("psi_p.csv", delimiter=",")
N = 10
D = 20
G_X = 40
G_R = 1

grid_size = (100, 100)
utl = utl(resolution=grid_size)
x_grid, y_grid, dx, dy = utl.grid_data

xy_mask, mask_idx = utl.pillar_mask((x_grid, y_grid), D, N, G_X)

# data = utl.add_mask(data, xy_mask, D, N, G_X, mask_with=np.NaN)

x_mapped, y_mapped = utl.parall2square(data[:, 0], data[:, 1], 1/N, D, G_X)
psi, p = data[:, 2], data[:, 3]
data1 = tuple([x_mapped, y_mapped, psi, p])

psi_interp = utl.interp2grid(x_mapped, y_mapped, psi, x_grid, y_grid, method='linear')
p_interp = utl.interp2grid(x_mapped, y_mapped, p, x_grid, y_grid, method='linear')

psi_interp = utl.insert_mask(psi_interp, mask_idx, mask_with=np.NaN)
p_interp = utl.insert_mask(p_interp, mask_idx, mask_with=np.NaN)

data2 = tuple([x_grid.flatten(), y_grid.flatten(),
               psi_interp.flatten(), p_interp.flatten()])

compare = False
if compare:
    utl.compare_plots(data1, data2)

u, v = utl.psi2uv(psi_interp, dx, dy, plot=True)

'''
x0 = 0
y0 = 0.5
point0 = np.array([x0, y0])
no_period = 1
stream = utl.simulate_particle(
    psi_interp, p_interp, start_point=point0, no_period=no_period)

# stream_original = stream
# for i in range(no_period):
#     stream_original[i][:, 0], stream_original[i][:, 1] = utl.square2parall(
#         stream[i][:, 0], stream[i][:, 1], 1/N, D, G_X)

utl.periodic_plot(stream)
'''