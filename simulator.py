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
pillar = utl.pillar(D)
pillars = utl.pillars(pillar, D, N, G_X)

xy_mask, mask_idx = utl.pillar_mask((x_grid, y_grid), pillars)




x_mapped, y_mapped = utl.parall2square(data[:, 0], data[:, 1], 1/N, D, G_X)
psi, p = data[:, 2], data[:, 3]
data1 = tuple([x_mapped, y_mapped, psi, p])

psi_interp = utl.interp2grid(
    x_mapped, y_mapped, psi, x_grid, y_grid, method='linear', recover=False)
p_interp = utl.interp2grid(
    x_mapped, y_mapped, p, x_grid, y_grid, method='linear', recover=False)

psi_interp = utl.insert_mask(psi_interp, mask_idx, mask_with=np.NaN)
p_interp = utl.insert_mask(p_interp, mask_idx, mask_with=np.NaN)

data2 = tuple([x_grid.flatten(), y_grid.flatten(),
               psi_interp.flatten(), p_interp.flatten()])

compare = False
if compare:
    utl.compare_plots(data1, data2)

u, v = utl.psi2uv(psi_interp, dx, dy, recover=True, plot=False)

x0 = 0
y0 = 16/60
point0 = np.array([x0, y0])
periods = 2

d_particle = 5/(D+G_X)
stream = utl.simulate_particle(
    d_particle, u, v, pillars, start_point=point0, periods=periods, plot=False)



a, b =utl.wallfunc((x_grid, y_grid), pillars)


