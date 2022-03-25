# Design automation procedure consist of getting the diameters of 
# particles for separation as input and propose dld design

import numpy as np
import matplotlib.pyplot as plt
from Conv_Base import DLD_Net
from DLD_env import DLD_env, Pillar
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
NN = DLD_Net()

d1 = 0.1
d2 = 0.5
mean_d = (d1+d2)/2

label_shape = (3,)
summary = False
NN.create_model(label_shape, summary)
NN.DLDNN.load_weights(NN.checkpoint_filepath)

def Surrogate_model(f, N, Re , NN):
    labels = np.array((f , N, Re))
    labels_Max = np.array((0.75, 10, 25)) 
    labels_norm = labels / labels_Max
    u_pred, v_pred = NN.DLDNN.predict(labels_norm[None, :])
    u_pred = u_pred[0, :, :, 0]
    v_pred = v_pred[0, :, :, 0]
    
    pillar = Pillar(f, N)
    dld = DLD_env(pillar, Re, resolution = NN.grid_size)

    uv_pred = (u_pred, v_pred)
    d_crt = NN.critical_dia(f, uv_pred, dld, 1, 0.01)
            
    return d_crt

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
        d_crt = Surrogate_model(x[0], x[1], x[2] , NN)
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

