# Design automation procedure consist of getting the diameters of
# particles for separation as input and propose dld design

import numpy as np
import copy
import matplotlib.pyplot as plt
from keras.models import load_model
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback
from pymoo.util.display import Display
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_reference_directions
from pymoo.optimize import minimize
import pandas as pd
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.factory import get_sampling, get_crossover, get_mutation

from DLD_Utils import DLD_Utils as utl
from Conv_Base import DLD_Net
from DLD_env import DLD_env, Pillar  

def optimizer(D1, D2, fc, Nc, Rec, fsr, n_objectives,
              n_varible, upper_lim, lower_lim, norm_vec,
              middle_peak, NeuralNet, n_gen, n_partitions, pop_size):

    
    

    

    # create the reference directions to be used for the optimization
    ref_dirs = get_reference_directions("das-dennis", n_objectives,
                                     n_partitions=n_partitions)

    class MyDisplay(Display):

        def _do(self, problem, evaluator, algorithm):
            super()._do(problem, evaluator, algorithm)
            idx = np.argmin(np.sum(algorithm.pop.get("F"), axis=1))
            self.output.append("d_critical", algorithm.pop.get("F")[idx][0])
            self.output.append("Re", algorithm.pop.get("F")[idx][1])
            self.output.append("N", algorithm.pop.get("F")[idx][2])
            self.output.append("Flexibility", algorithm.pop.get("F")[idx][3])
            self.output.append("Stability", algorithm.pop.get("F")[idx][4])
    # Callbacks

    class MyCallback(Callback):

        def __init__(self) -> None:
            super().__init__()
            self.data["best"] = []

        def notify(self, algorithm):
            self.data["best"].append(algorithm.pop.get("F").min())

    # create the algorithm object
    
    mask = ["real", "int", "real", "real"]

    sampling = MixedVariableSampling(mask, {
        "real": get_sampling("real_random"),
        "int": get_sampling("int_random")})

    crossover = MixedVariableCrossover(mask, {
        "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
        "int": get_crossover("int_sbx", prob=1.0, eta=3.0)
    })
    
    mutation = MixedVariableMutation(mask, {
        "real": get_mutation("real_pm", eta=3.0),
        "int": get_mutation("int_pm", eta=3.0)
    })

    # there is problem with mixed variable mutation when I applied it
    # the output becomes objects, it does not work with sampling either
    # sampling=sampling, crossover=crossover
    algorithm = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs, sampling=sampling,
                      crossover=crossover, mutation=mutation)
    
    class Problemwrapper(ElementwiseProblem):
        def __init__(self, D1, D2, norm_vec, n_objectives, n_varible,
                  upper_lim, lower_lim, middle_peak, fsr, Nc, Rec):
            super().__init__(n_var=n_varible,
                                n_obj=n_objectives,
                                xl=lower_lim,
                                xu=upper_lim)
            self.D1 = D1
            self.D2 = D2
            self.norm_vec = norm_vec
            self.lower_lim = lower_lim
            self.upper_lim = upper_lim
            self.middle_peak = middle_peak
            self.fsr = fsr
            self.Rec = Rec
            self.Nc = Nc
            self.fc = fc

            
        def _evaluate(self, x, out, *args, **kwargs):
            # Cuation: input in this section must be normilized
            f = x[0] / norm_vec[0]
            n = x[1] / norm_vec[1]
            re = x[2] / norm_vec[2]
            input = np.array((f, n, re))
            
            input = input[None, :]
            d_crt = DNN(input)
            
            # main opbjective
            f1 = 5 * np.abs(d_crt - (self.D1 + self.D2)/(2*x[3]))
            # Reynols objective
            if self.Rec == 'None':
                f2 = 0
            elif self.Rec == 'Max':
                f2 = 1 - re
            elif self.Rec == 'Min':
                f2 = re - self.lower_lim[2]/self.norm_vec[2]
            else:
                f2 = np.abs(re - self.Rec / self.norm_vec[2])
                
                
            # N objective 
            if self.Nc == 'None':
                f3 = 0
            elif self.Nc == 'Max':
                f3 = 1 - n
            elif self.Nc == 'Min':
                f3 = n - self.lower_lim[1]/self.norm_vec[1]
            else:
                f3 = np.abs(n - self.Nc / self.norm_vec[1])
            
            # F objective 
            if self.fc == 'None':
                f4 = 0
            elif self.fc == 'Max':
                f4 = 1 - f
            elif self.fc == 'Min':
                f4 = f - self.lower_lim[0]/self.norm_vec[0]
            else:
                f4 = np.abs(n - self.fc / self.norm_vec[0])
            
            
            # Conputing flexibility and stability index
            input1 = copy.deepcopy(input)
            input1[0][2] = self.upper_lim[2]/self.norm_vec[2]             
            
            input2 = copy.deepcopy(input)
            input2[0][2] = self.lower_lim[2]/self.norm_vec[2] 
            
            input3 = copy.deepcopy(input)
            input3[0][2] = self.middle_peak/self.norm_vec[2]             
            
            A1 = DNN(input1)
            A2 = DNN(input2)
            A3 = DNN(input3)
            
            bandwidth = max(A1, A2, A3) - min(A1, A2, A3)
            
            # Flexibility constraint
            f5 =  self.fsr * (1 - bandwidth)
            # Stability constraint 
            f6 = (1 - self.fsr) * (bandwidth)
            
            out['F'] = [f1, f2, f3, f4, f5, f6]

    Problem = Problemwrapper(D1, D2, norm_vec, n_objectives, n_varible, upper_lim, lower_lim,
     middle_peak, fsr, Nc, Rec)
    # execute the optimization
    res = minimize(Problem,
                   algorithm,
                   seed=1,
                   termination=('n_gen', n_gen),
                   callback=MyCallback(),
                   display=MyDisplay(),
                   verbose = True)


    idx = np.argmin(np.sum(res.F, axis=1))
    inp = res.X[idx][0:3] / norm_vec
    print(res.X[idx])
    d_opt = DNN(np.asarray(inp[None, :]).astype(np.float32))
    

    return np.asarray(res.X[idx]).astype(np.float32), d_opt


########################################################################
# Here we define the input for optimization#############################
########################################################################

fsr = 1 # Flexibility Stability dial 1 max flexibility 0 max stability
## For f, N and Re there is 4 options##
# None do not consider it 
# Max Maximum value
# Min Minimum value
# [Number] Specific value 
fc = 'None'
Nc = 'None'
Rec = 'Max'

D1 = 8 # Bigger particle diameter [um]
D2 = 5  # Smaller particle diameter [um]

n_objectives = 6 # Constant
n_varible = 4 # Constant 

# The dataset information
upper_lim = np.array([0.75, 10, 25, 10*(D1+D2)/6])
lower_lim = np.array([0.25, 3, 0.01, 1.1 * D1])
norm_vec = [0.75, 10, 25]
middle_peak = 5

#Loading Neural Network
DNN = load_model('DNN_model_hlayers8_nodes_128.h5')

# Do you want a plot of optimization results
plot = True
# Do you want to see the streamline of the two particles
streamline = True
num_per = 10 # Number of periods

# Find the optimum label for given input 
opt_label = np.zeros((len(norm_vec))) # Optimum results' labels
d_opt = 0 # Optimum critical diameter 

# Run optimization # Optimizr parameter are:
# n_gen  number of generation
# n_partitions number of reference partitions
# pop_size population population size
n_gen = 50
n_partitions = 5
pop_size = 260 # popultion size has lower value limit (write here)

opt_label , d_opt = optimizer(D1, D2, fc, Nc, Rec, fsr, n_objectives,
                              n_varible, upper_lim, lower_lim, norm_vec,
                              middle_peak, DNN, n_gen, n_partitions, pop_size)
d_mean = (D1 + D2)/2


if plot:
    re1 = np.linspace(0.01, 25, 30)
    input = np.column_stack((np.tile(opt_label[0:2],(30,1)), re1))
    d_crt1 = DNN(input/norm_vec)
    plt.figure()
    plt.plot(re1, d_crt1*opt_label[3])
    plt.plot(opt_label[2], d_opt*opt_label[3], 'ro')
    plt.plot([0.1, 25], [d_mean, d_mean], 'k')
    plt.legend(['Design_Range', 'Optimum_d_crt', 'Desired_d_crt'])
    plot_data = np.column_stack([re1, d_crt1])
    df = pd.DataFrame(plot_data)
    header_srt = ['Re' , 'D_crt']
    df.to_excel("plot_data.xlsx", header=header_srt, index=True)
    

if streamline:
    NN = DLD_Net()
    label_shape = opt_label[0:3].shape
    NN.create_model(label_shape, summary=False)
    NN.DLDNN.load_weights(NN.checkpoint_filepath)
    f =  opt_label[0]
    
    periods = num_per


    pillar = Pillar(opt_label[0], opt_label[1])
    NN.dld = DLD_env(pillar, opt_label[2], resolution=NN.grid_size)

    input = opt_label[0:3] / norm_vec
    u, v = NN.DLDNN.predict(input[None, :])
    u = u[0, :, :, 0]
    v = v[0, :, :, 0]
    uv = (u, v)
    
    dp = D1 / opt_label[3]
    start_point = (0, f/2+dp*(1-f)/2)
    s1, m1 = NN.dld.simulate_particle(dp*(1-f), uv, start_point, periods, plot=True)

    dp = D2 / opt_label[3]
    start_point = (0, f/2+dp*(1-f)/2)
    s1, m1 = NN.dld.simulate_particle(dp*(1-f), uv, start_point, periods, plot=True)



re2 = np.array((0.01, 5, 25))
input = np.column_stack((np.tile(opt_label[0:2],(3,1)), re2))
d_crt2 = DNN(input/norm_vec)
bandwidth = np.asarray((max(d_crt2) - min(d_crt2)) * opt_label[3]).astype(np.float32)

# Save the label 
opt_data = np.column_stack([opt_label.reshape((1,4)), D1, D2, d_mean,
                            d_opt*opt_label[3], bandwidth])

df = pd.DataFrame(opt_data)
header_srt = ['f', 'N', 'Re','G','d1', 'd2','d mean', 'd opt', 'bandwidth']
df.to_excel("opt_data.xlsx", header=header_srt, index=True)



