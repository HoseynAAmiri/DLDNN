import mph
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import Streamline_Particle as spt

'''
DataBase class handle variety of task from generation data
for neural network training to handeling information for
actual data and grid size and disterbution
'''


class DataBase():
    def __init__(self):
        '''
        The setting in which our database is created
        D is the diameter of pillars
        G is the gap between pilarrs
        N is the periodic number of pillars lattice
        Re is Reynols number
        '''
        self.D = np.linspace(10, 50, 10)
        self.G = np.linspace(10, 50, 10)
        self.N = np.linspace(2, 10, 10)
        self.Re = np.linspace(0.1, 50, 20)
        # size of input for NN or in some way our lable
        self.lable_size = 4
        self.train_ratio = 0.8
        self.grid_size = 101

        self.data_size = len(self.D)*len(self.G)*len(self.N)*len(self.Re)

        self.train_data_size = int(np.floor(0.8*self.data_size))
        self.test_data_size = self.data_size - self.train_data_size

        # Define the shape of the data sets similar to mnist datasets
        # DLD_Dataset = tuple((tuple((np.zeros((train_data_size,
        # grid_size,grid_size)),np.zeros((train_data_size,lable_size))))
        # ,tuple((np.zeros((test_data_size,grid_size,grid_size)),np.zeros((test_data_size,lable_size))))))

        # Create zero numpy array to create a dataset like mnist
        self.x_train = np.zeros(
            (self.train_data_size, self.grid_size, self.grid_size))
        self.y_train = np.zeros((self.train_data_size, self.lable_size))
        self.x_test = np.zeros(
            (self.test_data_size, self.grid_size, self.grid_size))
        self.y_test = np.zeros((self.train_data_size, self.lable_size))

        self.x_train = np.zeros(
            (self.train_data_size, self.grid_size, self.grid_size))
        self.y_train = np.zeros((self.train_data_size, self.lable_size))
        self.x_test = np.zeros(
            (self.test_data_size, self.grid_size, self.grid_size))
        self.y_test = np.zeros((self.train_data_size, self.lable_size))

        # Create grid
        xx = np.linspace(0, 1, self.grid_size)
        yy = np.linspace(0, 1, self.grid_size)
        self.x_grid, self.y_grid = np.meshgrid(xx, yy)

        '''
        This function get the data in parallelogram coordinate(data created in
        COMSOL) and by a shear transformation and scaling, transform it to
        a unitary square domain(which is used for NN training )
        '''

    def parall2square(self, x, y, slope, D, G_X, G_R=1):
        # Domain shear transformation from parallelogram to a rectangular
        x_mapped = x
        y_mapped = y - slope * x

        # Domain transformation from rectangular to unitariy square
        X_mapped_MAX = D + G_X
        Y_mapped_MAX = D + G_X * G_R

        x_mapped = x_mapped / X_mapped_MAX
        y_mapped = y_mapped / Y_mapped_MAX

        return x_mapped, y_mapped

    def square2parall(self, x, y, slope, D, G_X, G_R=1):
        X_MAX = D + G_X
        Y_MAX = D + G_X * G_R

        # Scaling square to rectangle
        x_mapped = x * X_MAX
        y_mapped = y * Y_MAX

        # Mapping rectangle to parallelogram by shear transformation
        x_mapped = x_mapped
        y_mapped = y_mapped + slope * x_mapped

        return x_mapped, y_mapped

    def interp2grid(self, x_mapped, y_mapped, data_mapped, x_grid, y_grid):
        # Interpolation of mapped data to x & y grid
        mapped = np.array([x_mapped, y_mapped]).T
        data_interp = griddata(mapped, data_mapped,
                               (x_grid, y_grid), method='nearest')

        return np.nan_to_num(data_interp)

    def mapping_plot(self, x, y, u, v, x_mapped, y_mapped, u_mapped, v_mapped):
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set_title("u (before)")
        ax1.set_aspect('equal')
        plt.scatter(x, y, c=u)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_title("v (before)")
        ax2.set_aspect('equal')
        plt.scatter(x, y, c=v)

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_title("u (after)")
        ax3.set_xlim([0, 1])
        ax3.set_ylim([0, 1])
        ax3.set_aspect('equal')
        plt.scatter(x_mapped, y_mapped, c=u_mapped)
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_title("v (after)")
        ax4.set_xlim([0, 1])
        ax4.set_ylim([0, 1])
        ax4.set_aspect('equal')
        plt.scatter(x_mapped, y_mapped, c=v_mapped)

        plt.show()

    def simulate_particle(self, u_interp, v_interp, start_point,
                          no_period=1, plot=False):

        shape = u_interp.shape
        xx = np.linspace(0, 1, shape[0])
        yy = np.linspace(0, 1, shape[1])
        x_grid, y_grid = np.meshgrid(xx, yy)

        s = []
        counter = np.zeros((no_period,))
        for i in range(no_period):
            s.append(spt.streamplot(x_grid, y_grid, u_interp,
                                    v_interp, start_point=start_point))
            if s[i][-1, 0] == 1:
                start_point = s[i][-1, :] - [1, 0]
            else:
                start_point = s[i][-1, :] + [0, 1]

                if i+1 < no_period:
                    counter[i+1] = 1

        shifts = np.cumsum(counter)

        if plot:
            fig = plt.figure(figsize=(6, 6))
            ax1 = fig.add_subplot(2, 2, 1)
            plt.xlim([0, 1])
            plt.ylim([0, 1])

            step_data = np.zeros((no_period, 2))
            for i in range(no_period):
                plt.plot(s[i][:, 0], s[i][:, 1], color=str(i/no_period))

                step_data[i] = [s[i][0, 1], s[i][-1, 1]]

            ax2 = fig.add_subplot(2, 2, 2)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.plot(step_data[:, 0], step_data[:, 1])

            ax3 = fig.add_subplot(2, 1, 2)

            plt.step(range(no_period),
                     (step_data[:, 1]-shifts)/abs(min(step_data[:, 1]-shifts)))

            plt.show()

        return s

    def generate(self):
        # Import Comsol model
        client = mph.start()
        pymodel = client.load('DLD_COMSOL.mph')
        self.model = pymodel.java

        ''' 
        model.param().set("Do", str(D) + "[um]")
        model.param().set("G", str(G) + "[um]")
        model.param().set("N", str(N))
        model.param().set("Re", str(Re))
        
        model.component("comp1").geom("geom1").run("fin");
        
        
        # Run the model
        model.study('std1').run()
        
        # Export the data
        model.result().export("data1").set("filename", "F:\\Microfluidic\\05-DLDNN\\Data_Base\\Data2.csv");
        model.result().export("data1").run();
        
        data = np.genfromtxt("data2.csv", delimiter=",")
        '''


m = DataBase()
data = np.genfromtxt("data4.csv", delimiter=",")
data = np.nan_to_num(data)
x_mapped, y_mapped = m.parall2square(data[:, 0], data[:, 1], 1/10, 20, 40)
u_mapped, v_mapped = m.parall2square(data[:, 2], data[:, 3], 1/10, 20, 40)

u_interp = m.interp2grid(x_mapped, y_mapped, u_mapped,
                         m.x_grid, m.y_grid)
v_interp = m.interp2grid(x_mapped, y_mapped, v_mapped,
                         m.x_grid, m.y_grid)


m.mapping_plot(x_mapped, y_mapped, u_mapped, v_mapped, m.x_grid.flatten(
 ), m.y_grid.flatten(), u_interp.flatten(), v_interp.flatten())


x0 = 0
y0 = 0.5
point0 = [x0, y0]
no_period = 20
s = m.simulate_particle(
    u_interp, v_interp, start_point=point0, no_period=no_period, plot=True)


fig = plt.figure(figsize=(6, 6))
ax = fig.gca()
ax.set_aspect('equal')

for i in range(no_period):
    x_original, y_original = m.square2parall(s[i][:, 0], s[i][:, 1], 1/10, 20, 40)
    plt.plot(x_original, y_original , color=str(i/no_period))

plt.show()
