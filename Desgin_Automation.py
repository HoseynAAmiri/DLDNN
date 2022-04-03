# Design automation procedure consist of getting the diameters of
# particles for separation as input and propose dld design

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback
from pymoo.util.display import Display
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_reference_directions
from pymoo.optimize import minimize
import pandas as pd
import time
from tqdm import tqdm
############################################
# here I want to choose to random number   #
# for d1 and d2 nad compare the d critical #
# to d critical of our numerical data      #
############################################


def optimizer(d1, d2):

  d1 = d1_ls[i]
  d2 = d2_ls[i]

  mean_d = (d1+d2)/2
  DNN = load_model('DNN_model_hlayers8_nodes_32.h5')

  # create the reference directions to be used for the optimization
  ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

  class MyDisplay(Display):

      def _do(self, problem, evaluator, algorithm):
          super()._do(problem, evaluator, algorithm)
          idx = np.argmin(np.sum(algorithm.pop.get("F"), axis=1))
          self.output.append("d_critical", algorithm.pop.get("F")[idx][0])
          self.output.append("Re", algorithm.pop.get("F")[idx][1])
          self.output.append("f", algorithm.pop.get("F")[idx][2])
  # Callbacks

  class MyCallback(Callback):

      def __init__(self) -> None:
          super().__init__()
          self.data["best"] = []

      def notify(self, algorithm):
          self.data["best"].append(algorithm.pop.get("F").min())

  # create the algorithm object
  algorithm = NSGA3(pop_size=100, ref_dirs=ref_dirs)

  class Problemwrapper(ElementwiseProblem):
      def __init__(self, mean_d, norm_vec):
          super().__init__(n_var=3,
                              n_obj=3,
                              xl=np.array([0.25, 3, 0.01]),
                              xu=np.array([0.75, 10, 25]))
          self.mean_d = mean_d
          self.norm_vec = norm_vec

      def _evaluate(self, x, out, *args, **kwargs):

          f = x[0] / norm_vec[0]
          n = x[1] / norm_vec[1]
          re = x[2] / norm_vec[2]
          input = np.array((f, n, re))
          input = input[None, :]
          d_crt = DNN(input)
          f1 = 5 * np.abs(d_crt - self.mean_d)
          f2 = 1 - re
          f3 = f - 0.25
          out['F'] = [f1, f2, f3]

  Problem = Problemwrapper(mean_d, norm_vec)


  # execute the optimization
  res = minimize(Problem,
                 algorithm,
                 seed=1,
                 termination=('n_gen', 100),
                 callback=MyCallback(),
                 display=MyDisplay(),
                 verbose = True)


  idx = np.argmin(np.sum(res.F, axis=1))
  inp = res.X[idx] / norm_vec
  opt_d = DNN(inp[None, :])

  return res.X[idx], mean_d, opt_d


d1 = 5
d2 = 8
test_size = 20
gr = np.array(np.round(np.linspace(1.1, 6, test_size), 2).tolist())
gap = np.array(gr * np.max((d1, d2)))
d1_ls = d1 / gap
d2_ls = d2 / gap

norm_vec = [0.75, 10, 25]
optimization = True

# Find the optimum label for given input 
opt_label = np.zeros((test_size, len(norm_vec)))
d_mean = np.zeros(test_size)
d_opt = np.zeros(test_size)
pbar = tqdm(total=test_size, position=0, leave=True)

for i in range(test_size):
  opt_label[i,:] , d_mean[i], d_opt[i] = optimizer(d1_ls[i], d1_ls[i])
  pbar.update(1)
  time.sleep(0.1)

# Save the label 
opt_data = np.column_stack([opt_label, d1_ls, d2_ls, d_mean.T, gap.T, gr.T, d_opt.T])
df = pd.DataFrame(opt_data)
header_srt = ['f', 'N', 'Re','d1', 'd2','d mean', 'gap', 'gap ratio' 'd opt']
df.to_excel("gap_data.xlsx", header=header_srt, index=True)
    

    
  




