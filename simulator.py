import numpy as np
from DLD_env import DLD_env, Pillar
from DLD_Utils import DLD_Utils as utl
utl = utl()

data = np.genfromtxt("40_3_20_1.csv", delimiter=",")
D = 40
N = 3
G_X = 20
G_R = 1
Re = 1
grid_size =(128, 128)

pillar = Pillar(D, N, G_X, G_R)
dld = DLD_env(pillar, Re, resolution=grid_size)

x_mapped, y_mapped = utl.parall2square(data[:, 0], data[:, 1], pillar)
x_mapped[x_mapped > 1 - dld.dx/2] = 1
y_mapped[y_mapped > 1 - dld.dy/2] = 1

psi, p = data[:, 2], data[:, 3]
data1 = tuple([x_mapped, y_mapped, psi])

psi_interp = utl.interp2grid(
    x_mapped, y_mapped, psi, dld.x_grid, dld.y_grid, method='linear', recover=True)
p_interp = utl.interp2grid(
    x_mapped, y_mapped, p, dld.x_grid, dld.y_grid, method='linear', recover=True)

data2 = tuple([dld.x_grid.flatten(), dld.y_grid.flatten(), psi_interp.flatten()])

compare = True
if compare:
    utl.compare_plots(data1, data2)

v, u = utl.gradient(psi_interp, -dld.dx*(D+G_X)*1e-6, dld.dy*(D+G_X*G_R)*1e-6)

'''
x0 = 0
y0 = 30/(D+G_X)
point0 = np.array([x0, y0])
periods = 20
d_particle = 10/(D+G_X)
stream = dld.simulate_particle(d_particle, (u, v), pillar.pillars, point0, periods=periods, plot=True)
'''
