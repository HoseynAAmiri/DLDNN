import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

z = [1, 2, 3, 4, 5]
for i, i1 in zip(range(len(z)), tqdm(range(len(z)))):
    time.sleep(0.1)
        
    for ii, i2 in zip(range(len(z)), tqdm(range(len(z)), leave=False)):
        
        time.sleep(1)