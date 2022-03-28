import numpy as np
import time
from tqdm import tqdm


from DLD_Utils import DLD_Utils as utl
from Conv_Base import DLD_Net
from DLD_env import DLD_env, Pillar
# load CNN 
NN = DLD_Net()
# Load data
dataset_name = "dataset2288"
dataset = utl.load_data(dataset_name)
D = np.zeros((len(dataset[2]),1))

pbar = tqdm(total=len(dataset[2]), position=0, leave=True)

for i in range(len(dataset[2])):
    
    # Compute D critical
    f, N, Re = dataset[2][i]  
    pillar = Pillar(f, N)
    dld = DLD_env(pillar, Re, resolution = NN.grid_size)
    uv = (dataset[0][i], dataset[1][i])
    d_crt = NN.critical_dia(f, uv, dld, 1, 0.01)
    
    # Create D critical dataset 
    D[i] = d_crt
    pbar.update(1)
    time.sleep(0.1)

# Create dataset
new_dataset = (dataset[2], D)
# Save the new dataset
utl.save_data(new_dataset, 'direct_dataset2288')