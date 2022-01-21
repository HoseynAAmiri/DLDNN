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

'''
data = np.genfromtxt("data4.csv", delimiter=",")
data = np.nan_to_num(data)
N = 10
D = 20
G_X = 40

x_mapped, y_mapped = m.parall2square(data[:, 0], data[:, 1], 1/N, D, G_X)
u_mapped, v_mapped = m.parall2square(data[:, 2], data[:, 3], 1/N, D, G_X)

u_interp = m.interp2grid(x_mapped, y_mapped, u_mapped,
                         m.x_grid, m.y_grid)
v_interp = m.interp2grid(x_mapped, y_mapped, v_mapped,
                         m.x_grid, m.y_grid)

compare = True

data1 = tuple([x_mapped, y_mapped, u_mapped, v_mapped])
data2 = tuple([m.x_grid.flatten(), m.y_grid.flatten(),
               u_interp.flatten(), v_interp.flatten()])
if compare:
    m.compare_plots(data1, data2)

x_original, y_original = m.square2parall(m.x_grid, m.y_grid, 1/N, D, G_X)
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

stream_original = stream
for i in range(no_period):
    stream_original[i][:, 0], stream_original[i][:, 1] = m.square2parall(
        stream[i][:, 0], stream[i][:, 1], 1/N, D, G_X)

m.periodic_plot(stream_original)
'''
