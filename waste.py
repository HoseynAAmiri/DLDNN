from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from math import atan

a = np.linspace(0, 1, 10)
b = a * a.shape[0]

print(b.astype(int), a.shape[0])