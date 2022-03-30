# Design automation procedure consist of getting the diameters of 
# particles for separation as input and propose dld design

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

d1 = 0.1
d2 = 0.5
mean_d = (d1+d2)/2

DNN = load_model('DNN_model_hlayers5_nodes_8.h5')
input = np.array((0.1, 0.1, 0.1))
input = input[None, :]
DNN.summary()
x = DNN(input)
print(x)
''''
# create the reference directions to be used for the optimization
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=5)

# Call backs
class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []

    def notify(self, algorithm):
        self.data["best"].append(algorithm.pop.get("F").min())

# create the algorithm object
algorithm = NSGA3(pop_size=30,
                  ref_dirs=ref_dirs)

class Problemwrapper(ElementwiseProblem):
    def __init__(self, mean_d):
        super().__init__(n_var=3,
                            n_obj=3,
                            xl=np.array([0.25, 3, 0.01]),
                            xu=np.array([0.75, 10, 25]))
        self.mean_d = mean_d

    def _evaluate(self, x, out, *args, **kwargs):
        d_crt = DNN(x[0], x[1], x[2] )
        f1 = np.abs(d_crt - self.mean_d)
        f2 = 1 / x[2]
        f3 = x[0]
        out['F'] = [f1, f2, f3]

Problem = Problemwrapper(mean_d)
# execute the optimization
res = minimize(Problem,
               algorithm,
               seed=1,
               termination=('n_gen', 20),
               callback=MyCallback(),
               verbose = True)

val = res.algorithm.callback.data["best"]
plt.plot(np.arange(len(val)), val)
plt.show()

Scatter().add(res.F).show()

'''