import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
from physical_neural_network import PINN
from DLD_Utils import DLD_Utils as utl
utl=utl()

N_train = 1_000
nIter = 10

layers = [6, 20, 2]

# Load Data
dataset = utl.load_data('dataset')

psi = dataset[0]  # L x N x N
pre = dataset[1]  # L x N x N
l = dataset[2]  # L x 4

N = psi[0].shape[0]
L = l.shape[0]

# Rearrange Data
xx = np.linspace(0, 1, N)
yy = np.linspace(0, 1, N)
x_grid, y_grid = np.meshgrid(xx, yy)  # N x N

XX = np.tile(np.array([x_grid.flatten()]), (1, L))  # N2 x L
YY = np.tile(np.array([y_grid.flatten()]), (1, L))  # N2 x L

DD = np.tile(np.array([l[:, 0]]), (N * N, 1))  # N2 x L
NN = np.tile(np.array([l[:, 1]]), (N * N, 1))  # N2 x L
GG = np.tile(np.array([l[:, 2]]), (N * N, 1))  # N2 x L
RR = np.tile(np.array([l[:, 3]]), (N * N, 1))  # N2 x L

SS = np.array(psi).reshape(L, N * N).T  # N2 x L
PP = np.array(pre).reshape(L, N * N).T  # N2 x L

x = XX.flatten()[:, None]  # N2L x 1
y = YY.flatten()[:, None]  # N2L x 1

d = DD.flatten()[:, None]  # N2L x 1
n = NN.flatten()[:, None]  # N2L x 1
g = PP.flatten()[:, None]  # N2L x 1
r = RR.flatten()[:, None]  # N2L x 1

s = SS.flatten()[:, None]  # N2L x 1
p = PP.flatten()[:, None]  # N2L x 1

######################################################################
######################## Noiseles Data ###############################
######################################################################
# Training Data
idx = np.random.choice(N * N * L, N_train, replace=False)
x_train = x[idx, :]
y_train = y[idx, :]

d_train = d[idx, :]
n_train = n[idx, :]
g_train = g[idx, :]
r_train = r[idx, :]

s_train = s[idx, :]
p_train = p[idx, :]

# Training
model = PINN(x_train, y_train, d_train, n_train,
                g_train, r_train, s_train, p_train, layers)

model.save_model(name='model2')
model.load_model(name='model')
print('a')
        
