from DLD_Util import DLD_Util as utl
import numpy as np

m = utl()
data_psi = np.genfromtxt("psi.csv", delimiter=",")
data_p = np.genfromtxt("p.csv", delimiter=",")

data = np.concatenate((data_psi, np.array([data_p[:, 2]]).T), axis=1)
x = data[:, 0]
y = data[:, 1]


data = np.nan_to_num(data)
N = 10
D = 20
G_X = 40

grid_size = (100, 100)
x_grid_size = grid_size[0]
y_grid_size = grid_size[1]

xx = np.linspace(0, 1, x_grid_size)
yy = np.linspace(0, 1, y_grid_size)
x_grid, y_grid = np.meshgrid(xx, yy)

x_mapped, y_mapped = m.parall2square(data[:, 0], data[:, 1], 1/N, D, G_X)
u_mapped, v_mapped = m.parall2square(data[:, 2], data[:, 3], 1/N, D, G_X)

u_interp = m.interp2grid(x_mapped, y_mapped, u_mapped, x_grid, y_grid)
v_interp = m.interp2grid(x_mapped, y_mapped, v_mapped, x_grid, y_grid)

print(u_interp)


compare = True

data1 = tuple([x_mapped, y_mapped, u_mapped, v_mapped])
data2 = tuple([x_grid.flatten(), y_grid.flatten(),
               u_interp.flatten(), v_interp.flatten()])
if compare:
    m.compare_plots(data1, data2)

'''
x_original, y_original = m.square2parall(x_grid, y_grid, 1/N, D, G_X)
u_original, v_original = m.square2parall(u_interp, v_interp, 1/N, D, G_X)

data3 = tuple([x_original, y_original, u_original, v_original])

if compare:
    m.compare_plots(data2, data3)

x0 = 0
y0 = 0.5
point0 = np.array([x0, y0])
no_period = 50
stream = m.simulate_particle(
    u_interp, v_interp, start_point=point0, no_period=no_period)

# stream_original = stream
# for i in range(no_period):
#     stream_original[i][:, 0], stream_original[i][:, 1] = m.square2parall(
#         stream[i][:, 0], stream[i][:, 1], 1/N, D, G_X)

m.periodic_plot(stream)
'''
