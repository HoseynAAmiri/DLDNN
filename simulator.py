import numpy as np
from DLD_env import DLD_env, Pillar
from DLD_Utils import DLD_Utils as utl
utl = utl()

data = np.genfromtxt("psi_p.csv", delimiter=",")
D = 20
N = 10
G_X = 40
G_R = 1
Re = 1
grid_size =(128, 128)

pillar = Pillar(D, N, G_X, G_R)
dld = DLD_env(pillar, Re, resolution=grid_size)

x_mapped, y_mapped = utl.parall2square(data[:, 0], data[:, 1], pillar)

psi, p = data[:, 2], data[:, 3]
data1 = tuple([x_mapped, y_mapped, psi])

psi_interp = utl.interp2grid(
    x_mapped, y_mapped, psi, dld.x_grid, dld.y_grid, method='linear', recover=True)
p_interp = utl.interp2grid(
    x_mapped, y_mapped, p, dld.x_grid, dld.y_grid, method='linear', recover=True)

data2 = tuple([dld.x_grid.flatten(), dld.y_grid.flatten(), psi_interp.flatten()])

compare = False
if compare:
    utl.compare_plots(data1, data2)

v, u = utl.gradient(psi_interp, -dld.dx, dld.dy)


x0 = 0
y0 = 30/(D+G_X)
point0 = np.array([x0, y0])
periods = 20
d_particle = 13/(D+G_X)
stream = dld.simulate_particle(d_particle, (u, v), point0, periods=periods, plot=True)

