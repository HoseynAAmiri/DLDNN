import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

z = [1, 2, 3]


for i, i1 in zip(range(len(z)-1), tqdm(range(len(z)-1), leave=True)):

    time.sleep(0.1)
    
    pbar2 = tqdm(total=len(z), leave=True)    
    for i2 in range(len(z)):
        pbar2.update(1)
        time.sleep(1)
    pbar2.close()    
        
